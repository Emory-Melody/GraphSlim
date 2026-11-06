# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Grid search over reduction-method hyperparameters with validation-only selection.

``--hpo`` turns one ``graphslim.train_all`` invocation into a resumable grid search:

1. the search space is read from ``--hpo_space``, else ``graphslim/configs/hpo/<method>.json``,
   else a template listing the method's current settings is written for the user to edit;
2. the space, the number of trials and a time estimate are printed and confirmed (``--hpo_yes``
   skips the prompt);
3. every trial reduces the graph with its settings and scores the reduced graph by the mean
   validation accuracy of ``--hpo_eval_runs`` GCN fits. Test scores are never computed;
4. each finished trial is appended to ``hpo/<method>/<dataset>_r<rate>/trials.jsonl`` under
   ``--save_path``, so an interrupted search resumes from the first unfinished trial;
5. the best trial is written to ``best.json`` and can be replayed with ``--config``.
"""

from copy import copy
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import click

SPACE_DIR = Path(__file__).resolve().parent / "configs" / "hpo"
# Identity, device and search bookkeeping are never searched.
FORBIDDEN = {
    "method",
    "dataset",
    "setting",
    "device",
    "gpu_id",
    "logger",
    "metric",
    "config",
    "save_path",
    "load_path",
    "seed",
    "hpo",
    "hpo_space",
    "hpo_eval_runs",
    "hpo_yes",
    "eval_whole",
    "run_eval",
    "run_reduction",
    "checkpoints",
    "verbose",
    "wandb",
    "wandb_project",
    "wandb_run_name",
}


def utcnow():
    """Get the current UTC timestamp for search provenance.

    :returns: ISO 8601 timestamp string including the UTC offset.
    """
    return datetime.now(timezone.utc).isoformat()


def search_dir(args):
    """Resolve the output directory for a method/dataset/reduction-rate search.

    :param args: Configuration namespace containing the options used by this operation.
    :returns: Absolute ``Path`` beneath ``args.save_path/hpo``.
    """
    return (
        Path(args.save_path).expanduser().resolve()
        / "hpo"
        / args.method
        / f"{args.dataset}_r{args.reduction_rate}"
    )


def template_space(args):
    """Build an editable one-point search grid from the current method settings.

    :param args: Configuration namespace containing the options used by this operation.
    :returns: Parameter-to-candidate-list mapping using eligible scalar configuration values.
    """
    keys = sorted(k for k in getattr(args, "_method_config_keys", set()) if not k.startswith("_"))
    space = {}
    for key in keys:
        value = getattr(args, key, None)
        if key not in FORBIDDEN and isinstance(value, (int, float, str, bool)):
            space[key] = [value]
    if not space:
        for key in ("lr", "epochs", "weight_decay", "dropout", "hidden"):
            if hasattr(args, key):
                space[key] = [getattr(args, key)]
    return space


def load_space(path):
    """Read and validate a JSON grid search space.

    Comment keys beginning with an underscore are ignored. Forbidden parameters,
    empty candidate lists, and non-object JSON values raise ``click.BadParameter``.

    :param path: Filesystem location used by this operation.
    :returns: Mapping from parameter names to nonempty candidate lists.
    """
    space = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise click.BadParameter("search space must be a JSON object", param_hint="--hpo_space")
    cleaned = {}
    for key, values in space.items():
        if key.startswith("_"):
            continue
        if key in FORBIDDEN:
            raise click.BadParameter(f"'{key}' cannot be searched", param_hint="--hpo_space")
        if not isinstance(values, list) or not values:
            raise click.BadParameter(
                f"'{key}' must map to a non-empty list", param_hint="--hpo_space"
            )
        cleaned[key] = values
    if not cleaned:
        raise click.BadParameter("search space is empty", param_hint="--hpo_space")
    return cleaned


def resolve_space(args):
    """Select an explicit, shipped, or saved search grid, creating a template when needed.

    :param args: Configuration namespace containing the options used by this operation.
    :returns: Tuple ``(space_path, candidate_mapping, template_created)``.
    """
    folder = search_dir(args)
    folder.mkdir(parents=True, exist_ok=True)
    if getattr(args, "hpo_space", None):
        path = Path(args.hpo_space).expanduser()
        return path, load_space(path), False
    shipped = SPACE_DIR / f"{args.method}.json"
    if shipped.is_file():
        return shipped, load_space(shipped), False
    path = folder / "space.json"
    if path.is_file():
        return path, load_space(path), False
    space = template_space(args)
    payload = {
        "_comment": (
            f"Search space for {args.method} on {args.dataset}. Each key maps to the list of values to try; "
            "every combination is one trial. Widen the lists, then rerun with --hpo."
        ),
        **space,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, space, True


def grid(space):
    """Enumerate the Cartesian product of parameter candidates in sorted key order.

    :param space: Mapping from searchable parameter names to candidate lists.
    :returns: List of parameter dictionaries, one per trial.
    """
    keys = sorted(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def box_table(rows, header):
    """Render search parameters as a bordered plain text table.

    :param rows: Table body rows.
    :param header: Column headings for the rendered table.
    :returns: Formatted table string.
    """
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    render = lambda r: "|" + "|".join(f" {str(v).ljust(w)} " for v, w in zip(r, widths)) + "|"
    return "\n".join([line, render(header), line, *[render(r) for r in rows], line])


def format_seconds(seconds):
    """Format an estimated duration in seconds, minutes, or hours.

    :param seconds: Duration in seconds, or ``None`` when no estimate is available.
    :returns: Duration text, or an unknown-duration message for missing or non-finite values.
    """
    if seconds is None or not math.isfinite(seconds):
        return "unknown until the first trial finishes"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


class TrialLog:
    """Append-only record of finished trials; resuming skips every trial already recorded."""

    def __init__(self, folder):
        """Initialize TrialLog and its configuration.

        :param folder: Filesystem location used by this operation.
        """
        self.path = folder / "trials.jsonl"
        self.records = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["trial"]] = record

    def append(self, record):
        """Append a trial record to JSONL and update the in-memory trial index.

        :param record: JSON-serializable trial record including its unique ``trial`` key.
        """
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.records[record["trial"]] = record

    def mean_seconds(self):
        """Average durations of completed trials only.

        :returns: Mean elapsed seconds, or ``None`` if no trial has completed.
        """
        done = [r["seconds"] for r in self.records.values() if r.get("status") == "complete"]
        return statistics.mean(done) if done else None


def describe(args, path, space, trials, log, created):
    """Describe a search grid, completed trials, validation selection rule, and estimated remaining time.

    :param args: Configuration namespace containing the options used by this operation.
    :param path: Filesystem location used by this operation.
    :param space: Mapping from searchable parameter names to candidate lists.
    :param trials: Enumerated trial parameter dictionaries.
    :param log: TrialLog containing previously recorded trial results.
    :param created: Whether the search space template was just created.
    :returns: Multiline search summary.
    """
    rows = [[key, ", ".join(str(v) for v in values), len(values)] for key, values in space.items()]
    remaining = len(trials) - len(log.records)
    per_trial = log.mean_seconds()
    lines = [
        f"HPO for {args.method} on {args.dataset} (rate {args.reduction_rate}, setting {args.setting})",
        f"search space: {path}"
        + ("  [template just written; widen it before searching]" if created else ""),
        box_table(rows, ["parameter", "values", "count"]),
        f"trials: {len(trials)} total, {len(log.records)} already recorded, {remaining} to run",
        f"selection: mean validation accuracy of {args.hpo_eval_runs} GCN fits per trial; no test scores are computed",
        f"estimated time: {format_seconds(per_trial * remaining if per_trial else None)}"
        + (f" ({format_seconds(per_trial)} per trial so far)" if per_trial else ""),
        f"records: {log.path}",
    ]
    return "\n".join(lines)


def apply(args, params):
    """Copy a configuration and apply trial parameters for validation-only evaluation.

    :param args: Configuration namespace containing the options used by this operation.
    :param params: Mapping of parameter names to replacement values.
    :returns: Trial configuration with HPO and experiment tracking disabled.
    """
    run_args = copy(args)
    for key, value in params.items():
        setattr(run_args, key, value)
    if "epochs" in params or "eval_interval" in params:
        run_args.checkpoints = list(range(-1, run_args.epochs + 1, run_args.eval_interval))
    run_args.validation_only = True
    run_args.hpo = False
    run_args.wandb = False
    return run_args


def reset_graph(graph):
    """Clear synthetic graph attributes and cached samplers before another trial.

    :param graph: Input graph object.
    """
    if hasattr(graph, "reset"):
        graph.reset()
    else:
        for key in ("feat_syn", "adj_syn", "labels_syn"):
            setattr(graph, key, None)


def validation_score(graph, run_args, runs):
    """Fit repeated downstream models and collect validation-only scores on a reduced graph.

    :param graph: Input graph object.
    :param run_args: Configuration namespace containing the options used by this operation.
    :param runs: Number of independently seeded downstream evaluations.
    :returns: List of validation scores; seeds increment from ``run_args.seed``.
    """
    from graphslim.dataset.utils import get_syn_data
    from graphslim.evaluation import Evaluator
    from graphslim.utils import seed_everything

    graph.feat_syn, graph.adj_syn, graph.labels_syn = get_syn_data(
        graph, run_args, model_type=run_args.eval_model, verbose=False
    )
    scores = []
    for index in range(runs):
        seed_everything(run_args.seed + index)
        evaluator = Evaluator(run_args)
        value = evaluator.test(
            graph, model_type=run_args.eval_model, verbose=False, reduced=True, validation_only=True
        )
        scores.append(float(value))
    return scores


def run_search(args, graph):
    """Run or resume a grid search selected only by mean validation accuracy.

    Trial records are appended to JSONL. The selected parameters and summary
    are saved beneath the search directory; no test metric selects the winner.

    :param args: Configuration namespace containing the options used by this operation.
    :param graph: Input graph object.
    :returns: Best completed trial record, or None if the search is declined or no trial completes.
    """
    from graphslim.reduction import create_reducer
    from graphslim.utils import seed_everything

    if getattr(args, "eval_whole", False):
        raise click.BadParameter(
            "--hpo searches reduction settings; it cannot run with --eval_whole"
        )
    folder = search_dir(args)
    path, space, created = resolve_space(args)
    trials = grid(space)
    log = TrialLog(folder)
    click.echo(describe(args, path, space, trials, log, created))
    if created and len(trials) == 1:
        click.echo(
            "The template holds a single point. Edit it to add values, then rerun with --hpo."
        )
        return None
    if not getattr(args, "hpo_yes", False) and not click.confirm(
        "Start the search?", default=False
    ):
        click.echo("HPO cancelled; nothing was run.")
        return None
    (folder / "space_used.json").write_text(json.dumps(space, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    finished_now = 0
    for trial, params in enumerate(trials):
        if trial in log.records:
            continue
        run_args = apply(args, params)
        run_args.save_path = str(folder / f"trial_{trial:03d}")
        Path(run_args.save_path).mkdir(parents=True, exist_ok=True)
        tick = time.monotonic()
        record = dict(trial=trial, params=params, utc=utcnow())
        try:
            seed_everything(args.seed)
            reset_graph(graph)
            reducer = create_reducer(args.method, setting=args.setting, data=graph, args=run_args)
            reducer.reduce(graph, verbose=False)
            scores = validation_score(graph, run_args, args.hpo_eval_runs)
            record.update(
                status="complete",
                validation=scores,
                validation_mean=statistics.mean(scores),
                reduced_nodes=int(graph.feat_syn.shape[0]),
                checkpoints=run_args.save_path,
            )
        except Exception as error:  # noqa: BLE001 - a bad combination must not stop the search
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}"[:500],
                traceback=traceback.format_exc()[-2000:],
            )
        record["seconds"] = round(time.monotonic() - tick, 1)
        log.append(record)
        finished_now += 1
        remaining = len(trials) - len(log.records)
        eta = log.mean_seconds() * remaining if log.mean_seconds() else None
        summary = (
            f"val {100 * record['validation_mean']:.2f}"
            if record["status"] == "complete"
            else record["error"]
        )
        click.echo(
            f"[{len(log.records)}/{len(trials)}] trial {trial} {params} -> {summary} "
            f"({record['seconds']:.0f} s, ETA {format_seconds(eta)})"
        )
    complete = [r for r in log.records.values() if r.get("status") == "complete"]
    if not complete:
        click.echo("No trial completed.")
        return None
    best = max(complete, key=lambda r: (r["validation_mean"], -r["trial"]))
    (folder / "best.json").write_text(json.dumps(best["params"], indent=2) + "\n", encoding="utf-8")
    (folder / "summary.json").write_text(
        json.dumps(
            dict(
                utc=utcnow(),
                method=args.method,
                dataset=args.dataset,
                reduction_rate=args.reduction_rate,
                space=space,
                trials=len(trials),
                complete=len(complete),
                failed=len(log.records) - len(complete),
                best_trial=best["trial"],
                best_params=best["params"],
                best_validation_mean=best["validation_mean"],
                selection="highest mean validation accuracy, earliest trial on ties; no test metric used",
                wall_seconds_this_session=round(time.monotonic() - started, 1),
                trials_run_this_session=finished_now,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    top = sorted(complete, key=lambda r: -r["validation_mean"])[:5]
    click.echo(
        box_table(
            [
                [r["trial"], f"{100 * r['validation_mean']:.2f}", json.dumps(r["params"])]
                for r in top
            ],
            ["trial", "val %", "params"],
        )
    )
    click.echo(
        f"best trial {best['trial']} written to {folder / 'best.json'}; replay with --config {folder / 'best.json'}"
    )
    return best

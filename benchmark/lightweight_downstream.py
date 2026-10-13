"""Auditable all-architecture / noise integration matrix for lightweight methods.

Smoke runs verify real data paths, not paper accuracy. Full runs retain method
defaults and the shared 300-step downstream protocol; they are not paper HPO.
Each reduction is saved once, reloaded independently for each architecture, and
all raw validation/test records and perturbation metadata are retained.
"""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("cgc", "clustgdd", "gcgp", "nope")
CONDITIONS = (
    "clean",
    "random_adj",
    "edge_drop",
    "random_feat",
    "feature_mask",
    "gaussian",
    "label_noise",
    "metattack",
)
MODELS = ("GCN", "SGC", "APPNP", "Cheby", "GraphSage", "GAT", "SGFormer")


def science_files():
    return sorted(
        [
            path
            for path in (ROOT / "graphslim").rglob("*")
            if path.suffix in (".py", ".json", ".npy", ".sh")
        ]
        + [
            Path(__file__),
            ROOT / "benchmark/__init__.py",
            ROOT / "benchmark/verify_lightweight.py",
            ROOT / "benchmark/rerun_nr.py",
            ROOT / "benchmark/reproduce.py",
            ROOT / "pyproject.toml",
        ]
    )


def digest():
    result = hashlib.sha256()
    for path in science_files():
        result.update(str(path.relative_to(ROOT)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def plan(options):
    return [
        {
            "dataset": dataset,
            "method": method,
            "noise": noise,
            "seed": seed,
            "id": f"{dataset}__{method}__{noise}__s{seed}",
        }
        for dataset in options.datasets
        for method in options.methods
        for noise in options.conditions
        for seed in options.seeds
    ]


def initialize(options):
    from benchmark.rerun_nr import utcnow, write_json

    folder = options.run_dir
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = {
        "version": 1,
        "created_utc": utcnow(),
        "source_sha256": digest(),
        "profile": options.profile,
        "conditions": options.conditions,
        "rate": options.rate,
        "noise_rate": options.noise_rate,
        "eval_seeds": options.eval_seeds,
        "models": list(MODELS),
        "jobs": plan(options),
        "purpose": "integration validation, not a paper reproduction or HPO score",
        "device": "cpu",
        "threads": options.threads,
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "source_sha256",
            "profile",
            "jobs",
            "rate",
            "noise_rate",
            "eval_seeds",
            "threads",
        ):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"Frozen manifest differs in {key}; use a new run directory")
        return previous
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in science_files():
            archive.write(path, path.relative_to(ROOT))
    write_json(manifest_path, manifest)
    return manifest


def export(folder, manifest):
    """A numeric cell requires a complete, same-source model/seed roster."""
    from benchmark.rerun_nr import write_json

    rows = []
    complete = failures = 0
    for job in manifest["jobs"]:
        path = folder / "jobs" / job["id"]
        result_path = path / "result.json"
        if not result_path.exists():
            if (path / "error.json").exists():
                failures += 1
            continue
        record = json.loads(result_path.read_text())
        if record["source_sha256"] != manifest["source_sha256"] or record["job"] != job:
            raise RuntimeError("Mixed source/job result")
        evaluations = record["evaluations"]
        expected = {
            (model, seed) for model in manifest["models"] for seed in manifest["eval_seeds"]
        }
        if {(value["model"], value["seed"]) for value in evaluations} != expected or len(
            evaluations
        ) != len(expected):
            raise RuntimeError("Missing/duplicate model or evaluation seed")
        complete += 1
        for model in manifest["models"]:
            scores = [value["test"] for value in evaluations if value["model"] == model]
            rows.append(
                {
                    **job,
                    "model": model,
                    "mean": statistics.mean(scores),
                    "std": statistics.pstdev(scores),
                    "n": len(scores),
                    "nodes": record["reduction"]["synthetic_nodes"],
                    "source_sha256": manifest["source_sha256"],
                }
            )
    fields = [
        "dataset",
        "method",
        "noise",
        "seed",
        "id",
        "model",
        "mean",
        "std",
        "n",
        "nodes",
        "source_sha256",
    ]
    with (folder / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "complete": complete,
        "failed": failures,
        "expected": len(manifest["jobs"]),
        "model_cells": len(rows),
        "profile": manifest["profile"],
        "source_sha256": manifest["source_sha256"],
        "rows": rows,
    }
    write_json(folder / "summary.json", summary)
    budget = (
        "Smoke: 20 reduction steps (ClustGDD also 20 pretrain steps), 20 downstream steps, hidden 32."
        if manifest["profile"] == "smoke"
        else "Full: method-specific reduction defaults, 300 downstream steps, hidden 256."
    )
    lines = [
        "### Lightweight methods: real-data integration checks",
        "",
        f"Profile: `{manifest['profile']}`; {complete}/{len(manifest['jobs'])} complete reductions, {len(rows)} model cells, {failures} failures.",
        "",
        "These are integration checks, NOT paper results or validation-selected HPO. "
        + budget
        + " One reduction/evaluation seed by default. Feature preprocessing follows each method, so do not rank these rows as a controlled comparison.",
        "",
        "| Dataset | Method | Noise | Reduction seed | Actual nodes | " + " | ".join(MODELS) + " |",
        "| --- | --- | --- | --- | --- | " + " | ".join(["---"] * len(MODELS)) + " |",
    ]
    for job in manifest["jobs"]:
        selected = [row for row in rows if row["id"] == job["id"]]
        by_model = {row["model"]: row for row in selected}
        values = [
            f"{100 * by_model[model]['mean']:.2f}" if model in by_model else "PENDING"
            for model in MODELS
        ]
        nodes = selected[0]["nodes"] if selected else "—"
        lines.append(
            f"| {job['dataset']} | {job['method']} | {job['noise']} | {job['seed']} | {nodes} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Cells are test accuracy (%) or the dataset metric; uncertainty/raw seeds are in `scores.csv` and `jobs/*/result.json`. No numeric partial cells. `metattack` is the package's PRBCD path, with 2 surrogate/attack steps in smoke mode, not literal MetaAttack. Clean jobs additionally retain fixed-model inference-noise results, two-candidate validation-only NAS, graph properties and oracle-threshold MIA diagnostics. NOPE preserves components and removes unlabeled supernodes; actual node counts can differ from the request.",
            "",
        ]
    )
    (folder / "results.md").write_text("\n".join(lines))
    return summary


def execute(options, manifest):
    from benchmark.rerun_nr import write_json, array_hash, serializable, lock_file
    from graphslim.config import get_args
    from graphslim.dataset import get_dataset, get_syn_data, perturb_graph, attack
    from graphslim.evaluation import Evaluator, NasEvaluator, PropertyEvaluator
    from graphslim.reduction import create_reducer
    from graphslim.utils import seed_everything
    from benchmark.verify_lightweight import verify
    import torch

    torch.set_num_threads(options.threads)
    write_json(options.run_dir / "author_parity.json", verify(options.references))
    with lock_file(options.run_dir / "manager.lock", blocking=False):
        for job in manifest["jobs"]:
            folder = options.run_dir / "jobs" / job["id"]
            if (folder / "result.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            if digest() != manifest["source_sha256"]:
                raise RuntimeError("Scientific source changed after planning")
            try:
                argv = [
                    "-D",
                    job["dataset"],
                    "-M",
                    job["method"],
                    "-R",
                    str(options.rate),
                    "-G",
                    "-1",
                    "-S",
                    str(job["seed"]),
                    "--run_eval",
                    "1",
                    "--save_path",
                    str(folder / "checkpoints"),
                ]
                if options.profile == "smoke":
                    argv += ["--epochs", "20", "--eval_epochs", "20", "--hidden", "32"]
                args = get_args(argv)
                if options.profile == "smoke":
                    args.clustgdd_pre_epochs = 20
                    args.clustgdd_hidden = 32
                    args.trans_num_layers = 1
                    args.attack_epochs = args.attack_pretrain_epochs = 2
                    args.attack_search_space, args.attack_final_samples = 4096, 2
                args.eval_mode = "cross"
                seed_everything(job["seed"])
                graph = get_dataset(args.dataset, args, str(ROOT / "data"))
                hashes = {
                    key: array_hash(getattr(graph, key))
                    for key in ("feat_full", "labels_train", "idx_train", "idx_val", "idx_test")
                }
                if job["noise"] == "metattack":
                    args.attack, args.ptb_r = "metattack", options.noise_rate
                    graph = attack(graph, args)
                    args.attack = None
                else:
                    graph = perturb_graph(
                        graph,
                        job["noise"],
                        0 if job["noise"] == "clean" else options.noise_rate,
                        job["seed"],
                        setting=args.setting,
                    )
                write_json(
                    folder / "config.json",
                    {key: serializable(value) for key, value in vars(args).items()},
                )
                graph = create_reducer(job["method"], args.setting, graph, args).reduce(
                    graph, verbose=False
                )
                properties = PropertyEvaluator(args).evaluate(graph)
                record = {
                    "job": job,
                    "source_sha256": manifest["source_sha256"],
                    "data_hashes": hashes,
                    "setting": args.setting,
                    "noise": graph.noise_metadata,
                    "reduction": graph.reduction_metadata,
                    "properties": properties,
                    "evaluations": [],
                    "inference_noise": [],
                }
                for model in manifest["models"]:
                    for eval_seed in manifest["eval_seeds"]:
                        model_args = deepcopy(args)
                        model_args.seed = eval_seed
                        evaluated = deepcopy(graph)
                        evaluated.feat_syn, evaluated.adj_syn, evaluated.labels_syn = get_syn_data(
                            evaluated, args, model_type=model
                        )
                        seed_everything(eval_seed)
                        evaluator = Evaluator(model_args)
                        validation, test, mia = evaluator.test(
                            evaluated, model, verbose=False, reduced=True, mode="cross", MIA=True
                        )
                        raw = {
                            "model": model,
                            "seed": eval_seed,
                            "validation": float(validation),
                            "test": float(test),
                            "oracle_mia": float(mia),
                        }
                        record["evaluations"].append(raw)
                        write_json(folder / "evaluations" / f"{model}__s{eval_seed}.json", raw)
                        if job["noise"] == "clean":
                            for noise in CONDITIONS[1:-2]:
                                corrupted = perturb_graph(
                                    evaluated,
                                    noise,
                                    options.noise_rate,
                                    job["seed"],
                                    setting=args.setting,
                                    stage="inference",
                                )
                                prefix = "full" if args.setting == "trans" else "test"
                                with torch.no_grad():
                                    prediction = evaluator.last_model.predict(
                                        getattr(corrupted, f"feat_{prefix}"),
                                        getattr(corrupted, f"adj_{prefix}"),
                                    )
                                prediction = (
                                    prediction[corrupted.idx_test]
                                    if args.setting == "trans"
                                    else prediction
                                )
                                score = float(
                                    args.metric(prediction, corrupted.labels_test.to(args.device))
                                )
                                record["inference_noise"].append(
                                    {
                                        "model": model,
                                        "seed": eval_seed,
                                        "noise": noise,
                                        "test": score,
                                        "checkpoint_retrained": False,
                                    }
                                )
                if job["noise"] == "clean":
                    nas_args = deepcopy(args)
                    nas_args.nas_epochs = args.eval_epochs
                    nas = NasEvaluator(nas_args)
                    nas.parameter_combinations = [
                        (2, args.hidden, 0.1, "relu"),
                        (2, args.hidden, 0.2, "linear"),
                    ]
                    nas.evaluate_ori(deepcopy(graph))
                    nas.evaluate_syn(deepcopy(graph))
                    record["nas"] = {
                        "purpose": "two-candidate entrypoint check, validation-only, no test selection",
                        "original_validation": nas.results_ori,
                        "synthetic_validation": nas.results_syn,
                        "selected_original": nas.best_params_ori,
                        "selected_synthetic": nas.best_params_syn,
                    }
                record["checkpoint_sha256"] = {
                    str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (folder / "checkpoints/reduced_graph").rglob("*.pt")
                }
                write_json(folder / "result.json", record)
                print(f"Completed {job['id']}: {len(record['evaluations'])} fits", flush=True)
            except Exception:
                write_json(
                    folder / "error.json",
                    {
                        "job": job,
                        "source_sha256": manifest["source_sha256"],
                        "error": traceback.format_exc(),
                    },
                )
                print(f"FAILED {job['id']}: see error.json", flush=True)
            summary = export(options.run_dir, manifest)
            write_json(
                options.run_dir / "heartbeat.json",
                {
                    "pid": os.getpid(),
                    "complete": summary["complete"],
                    "failed": summary["failed"],
                    "expected": summary["expected"],
                },
            )
    return export(options.run_dir, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--noise-rate", type=float, default=0.1)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=ROOT / "runs/reference")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    options.run_dir, options.references = options.run_dir.resolve(), options.references.resolve()
    if not 0 < options.rate <= 1 or not 0 <= options.noise_rate <= 1 or options.threads < 1:
        parser.error("Require 0 < rate <= 1, 0 <= noise-rate <= 1 and positive threads")
    if options.dry_run:
        print(
            f"{len(plan(options))} reductions; {len(plan(options)) * len(MODELS) * len(options.eval_seeds)} downstream fits; CPU profile={options.profile}"
        )
        return
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(options.threads)
    manifest = initialize(options)
    if not options.worker:
        # Isolate science before dispatch, so package development can continue.
        frozen = options.run_dir / "frozen_workspace"
        frozen.mkdir(exist_ok=True)
        with zipfile.ZipFile(options.run_dir / "source.zip") as archive:
            archive.extractall(frozen)
        for name in ("data", "runs"):
            link = frozen / name
            if not link.is_symlink():
                link.symlink_to(ROOT / name, target_is_directory=True)
        command = [sys.executable, "-B", "-m", "benchmark.lightweight_downstream"] + (
            list(argv) if argv is not None else sys.argv[1:]
        )
        command += [
            "--worker",
            "--run-dir",
            str(options.run_dir),
            "--references",
            str(options.references),
        ]
        os.environ["PYTHONPATH"] = str(frozen)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.chdir(frozen)
        os.execvpe(command[0], command, os.environ)
    summary = execute(options, manifest)
    print(
        f"Terminal: {summary['complete']}/{summary['expected']} complete, {summary['failed']} failed"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

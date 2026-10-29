"""Reproduce the GC4NC Table 9 performance matrix with this package.

The protocol follows GC4NC (arXiv:2406.16715v3, Appendix A.3). Each reduced graph
trains a two-layer GCN with 256 hidden units for 300 epochs, ten times. Reduction
checkpoints are selected on validation accuracy only. Condensation, coarsening and
sparsification methods run one reduction (seed 1). Random, K-Center and Herding
run three reductions (seeds 1, 2, 3), as stated in the paper.

Jobs run from a frozen copy of the package, so concurrent development cannot mix
source versions inside one batch. Several small jobs may share one CUDA device;
``--capacity`` bounds the summed dataset weights per device.

Usage::

    .venv-gpu/bin/python -m benchmark.reproduce_table9 plan --run-dir runs/table9_NEW
    .venv-gpu/bin/python -m benchmark.reproduce_table9 run --run-dir runs/table9_NEW \
        --devices MIG-... MIG-... --capacity 3
    .venv-gpu/bin/python -m benchmark.reproduce_table9 status --run-dir runs/table9_NEW
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

from benchmark import rerun_nr as nr
from benchmark.reproduce import RATES

ROOT = Path(__file__).resolve().parents[1]
DATA_SEED = 1
EVAL_RUNS = 10

# Package name -> name printed in GC4NC Table 9 (or the method's own name).
TABLE9 = {
    "cent_d": "Cent-D",
    "cent_p": "Cent-P",
    "random": "Random",
    "herding": "Herding",
    "kcenter": "K-Center",
    "t_spanner": "TSpanner",
    "averaging": "Averaging",
    "variation_neighborhoods": "VN",
    "vng": "VNG",
    "geom": "GEOM",
    "sfgc": "SFGC",
    "gcsntk": "GCSNTK",
    "gcdmx": "GCDMX",
    "gcondx": "GCondX",
    "gcond": "GCond",
    "doscond": "DosCond",
    "msgc": "MSGC",
    "sgdd": "SGDD",
    "gdem": "GDEM",
}
EXTRA = {
    "clustering": "Clustering",
    "doscondx": "DosCondX",
    "gcdm": "GCDM",
    "gecc": "GECC",
}
NAMES = {**TABLE9, **EXTRA, "whole": "Whole"}
FAMILY = {
    **{m: "Coreset / sparsification" for m in ("cent_d", "cent_p", "random", "herding", "kcenter", "t_spanner")},
    **{m: "Coarsening" for m in ("averaging", "variation_neighborhoods", "vng", "clustering")},
}
THREE_SEEDS = ("random", "kcenter", "herding")
# Rough memory/compute weight used to pack jobs on one device.
WEIGHT = {"cora": 1, "citeseer": 1, "pubmed": 1, "flickr": 2, "ogbn-arxiv": 2, "yelp": 2, "reddit": 3}
SMALL = ("cora", "citeseer", "pubmed")
# Expected relative cost, used to start long large-graph jobs first.
COST = {"geom": 9, "sfgc": 9, "gcond": 8, "doscond": 7, "doscondx": 6, "sgdd": 7, "msgc": 6,
        "gcsntk": 5, "gcondx": 5, "gdem": 5, "gcdmx": 3, "gcdm": 3}
# Paper Table 9 percentage labels for each CLI rate.
LABELS = {
    "citeseer": ("0.36", "0.90", "1.80"),
    "cora": ("0.50", "1.30", "2.60"),
    "pubmed": ("0.02", "0.03", "0.15"),
    "ogbn-arxiv": ("0.05", "0.25", "0.50"),
    "flickr": ("0.10", "0.50", "1.00"),
    "reddit": ("0.05", "0.10", "0.20"),
    "yelp": ("0.05", "0.10", "0.20"),
}
CSV_NAMES = {
    "citeseer": "Citeseer", "cora": "Cora", "pubmed": "Pubmed", "ogbn-arxiv": "Arxiv",
    "flickr": "Flickr", "reddit": "Reddit", "yelp": "Yelp",
}


def default_methods():
    return [*TABLE9, *EXTRA]


def plan_jobs(datasets, methods):
    jobs = []
    for dataset in datasets:
        jobs.append(dict(id=f"{dataset}__whole", dataset=dataset, method="whole", rate=None, seed=1))
        for method in methods:
            for rate in RATES[dataset]:
                for seed in (1, 2, 3) if method in THREE_SEEDS else (1,):
                    jobs.append(
                        dict(
                            id=f"{dataset}__{method}__r{rate}__s{seed}",
                            dataset=dataset,
                            method=method,
                            rate=rate,
                            seed=seed,
                        )
                    )

    def order(job):
        small = job["dataset"] in SMALL
        return (
            not small,
            # Small graphs: grouped by dataset for early feedback.
            datasets.index(job["dataset"]) if small else 0,
            # Large graphs: longest expected jobs first, then by dataset weight.
            0 if small else -COST.get(job["method"], 1) * WEIGHT[job["dataset"]],
            job["method"] in ("geom", "sfgc"),
            job["id"],
        )

    return sorted(jobs, key=order)


def science_files():
    files = set(nr.source_files())
    files.update(
        ROOT / name
        for name in ("benchmark/__init__.py", "benchmark/reproduce_table9.py", "benchmark/rerun_nr.py")
    )
    return sorted(files)


def initialize(options):
    folder = options.run_dir
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return nr.read_json(manifest_path)
    folder.mkdir(parents=True, exist_ok=True)
    datasets = list(options.datasets or RATES)
    methods = list(options.methods or default_methods())
    digest = hashlib.sha256()
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in science_files():
            name = str(path.relative_to(ROOT))
            content = path.read_bytes()
            digest.update(name.encode())
            digest.update(content)
            archive.writestr(name, content)
    frozen = folder / "frozen_workspace"
    frozen.mkdir(exist_ok=True)
    with zipfile.ZipFile(folder / "source.zip") as archive:
        archive.extractall(frozen)
    (frozen / "data").symlink_to(ROOT / "data", target_is_directory=True)
    manifest = dict(
        created_utc=nr.utcnow(),
        source_sha256=digest.hexdigest(),
        archive_sha256=hashlib.sha256((folder / "source.zip").read_bytes()).hexdigest(),
        upstream_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        protocol=dict(
            source="GC4NC arXiv:2406.16715v3 Appendix A.3",
            eval_model="GCN",
            hidden=256,
            nlayers=2,
            eval_epochs=300,
            run_eval=EVAL_RUNS,
            run_inter_eval=3,
            data_seed=DATA_SEED,
            reduction_seeds={"random/kcenter/herding": [1, 2, 3], "others": [1]},
            checkpoint_selection="validation accuracy only",
            uncertainty="population standard deviation over all test scores of a cell",
            rates="CLI fractions of labeled training nodes; Table 9 labels in LABELS",
        ),
        datasets=datasets,
        methods=methods,
        jobs=plan_jobs(datasets, methods),
    )
    nr.write_json(manifest_path, manifest)
    return manifest


def job_dir(folder, job):
    return folder / "jobs" / job["id"]


def state(folder, job):
    path = job_dir(folder, job)
    if (path / "result.json").exists():
        return "complete"
    if (path / "error.json").exists():
        return "failed"
    if (path / "launch.json").exists():
        launch = nr.read_json(path / "launch.json")
        try:
            command = Path(f"/proc/{launch['pid']}/cmdline").read_bytes()
            if job["id"].encode() in command:
                return "running"
        except (FileNotFoundError, ProcessLookupError):
            pass
        return "interrupted"
    return "pending"


def worker_arguments(job, folder):
    arguments = ["-D", job["dataset"], "-G", "0", "--seed", str(job["seed"])]
    arguments += ["--load_path", str(ROOT / "data")]
    arguments += ["--save_path", str(job_dir(folder, job) / "checkpoints")]
    arguments += ["--run_eval", str(EVAL_RUNS), "--eval_epochs", "300"]
    arguments += ["--eval_model", "GCN", "--hidden", "256", "--nlayers", "2"]
    if job["method"] == "whole":
        return arguments + ["-M", "random", "-W"]
    return arguments + ["-M", job["method"], "-R", job["rate"], "--run_reduction", "1"]


def execute(folder, job_id):
    manifest = nr.read_json(folder / "manifest.json")
    job = next(job for job in manifest["jobs"] if job["id"] == job_id)
    path = job_dir(folder, job)
    path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        import torch

        import graphslim
        from graphslim.config import get_args
        from graphslim.dataset import get_dataset
        from graphslim.train_all import run
        from graphslim.utils import seed_everything

        frozen = (folder / "frozen_workspace").resolve()
        if not Path(graphslim.__file__).resolve().is_relative_to(frozen):
            raise RuntimeError(f"Imported live package {graphslim.__file__}, not the frozen copy")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each job requires exactly one masked CUDA device.")
        # GEOM/SFGC write expert buffers relative to the working directory.
        work = path / "work"
        work.mkdir(exist_ok=True)
        os.chdir(work)
        args = get_args(worker_arguments(job, folder))
        nr.write_json(path / "config.json", {k: nr.serializable(v) for k, v in vars(args).items()})
        data_record = {}

        def loader(name, config, load_path):
            with nr.lock_file(ROOT / "data" / f".{name}.prepare.lock"):
                seed_everything(DATA_SEED)
                graph = get_dataset(name, config, load_path)
            data_record.update(
                nodes=graph.num_nodes,
                classes=int(graph.nclass),
                split_sizes={
                    split: int(getattr(graph, f"idx_{split}").numel())
                    for split in ("train", "val", "test")
                },
                test_sha256=nr.array_hash(graph.idx_test),
                labels_sha256=nr.array_hash(graph.labels_full),
            )
            return graph

        results = []

        def record(config, graph, evaluator, result):
            if len(evaluator.last_results) != EVAL_RUNS:
                raise RuntimeError("Incomplete downstream evaluation")
            nodes = None
            if job["method"] != "whole":
                feat = getattr(graph, "feat_syn", None)
                if feat is None or not torch.isfinite(torch.as_tensor(feat)).all():
                    raise RuntimeError("Missing or non-finite reduced features")
                nodes = int(feat.shape[0])
            results.append(
                dict(
                    evaluations=evaluator.last_results,
                    mean=float(result[0]),
                    std=float(result[1]),
                    reduced_nodes=nodes,
                )
            )

        run(args, load_dataset=loader, on_result=record)
        if len(results) != 1:
            raise RuntimeError(f"Expected one evaluated graph, got {len(results)}")
        checkpoints = {
            str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted((path / "checkpoints").rglob("*.pt"))
        }
        nr.write_json(
            path / "result.json",
            dict(
                job=job,
                completed_utc=nr.utcnow(),
                elapsed_seconds=time.monotonic() - started,
                source_sha256=manifest["source_sha256"],
                package_file=graphslim.__file__,
                device=torch.cuda.get_device_name(0),
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                data=data_record,
                checkpoint_sha256=checkpoints,
                **results[0],
            ),
        )
    except BaseException as exc:  # noqa: BLE001 - every outcome is recorded
        message = str(exc)
        out_of_memory = "out of memory" in message.lower() or isinstance(exc, MemoryError)
        nr.write_json(
            path / "error.json",
            dict(
                error=message[:4000],
                kind="OOM" if out_of_memory else type(exc).__name__,
                traceback=traceback.format_exc()[-20000:],
                elapsed_seconds=time.monotonic() - started,
                utc=nr.utcnow(),
            ),
        )
        raise


def manage(options, manifest):
    folder = options.run_dir
    frozen = folder / "frozen_workspace"
    capacity = {device: options.capacity for device in options.devices}
    active = {}
    with nr.lock_file(folder / "manager.lock", blocking=False):
        while True:
            for job_id, item in list(active.items()):
                process = item["process"]
                elapsed = time.monotonic() - item["started"]
                if process.poll() is None and elapsed > options.timeout:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    nr.write_json(
                        job_dir(folder, item["job"]) / "error.json",
                        dict(error=f"Timeout after {options.timeout} s", kind="TIMEOUT", utc=nr.utcnow()),
                    )
                if process.poll() is None:
                    continue
                if not any(
                    (job_dir(folder, item["job"]) / name).exists()
                    for name in ("result.json", "error.json")
                ):
                    nr.write_json(
                        job_dir(folder, item["job"]) / "error.json",
                        dict(
                            error="Worker exited without a result",
                            kind="CRASH",
                            exit_code=process.returncode,
                            utc=nr.utcnow(),
                        ),
                    )
                item["log"].close()
                del active[job_id]
                print(f"{nr.utcnow()} finished {job_id} exit={process.returncode}", flush=True)
            states = {job["id"]: state(folder, job) for job in manifest["jobs"]}
            pending = [
                job
                for job in manifest["jobs"]
                if states[job["id"]] in ("pending", "interrupted") and job["id"] not in active
            ]
            used = Counter()
            for item in active.values():
                used[item["device"]] += WEIGHT[item["job"]["dataset"]]
            exclusive = {
                (item["job"]["dataset"], item["job"]["method"])
                for item in active.values()
                if item["job"]["method"] in ("geom", "sfgc")
            }
            for job in list(pending):
                weight = WEIGHT[job["dataset"]]
                if (job["dataset"], job["method"]) in exclusive:
                    continue
                device = next(
                    (d for d in options.devices if used[d] + weight <= capacity[d]), None
                )
                if device is None:
                    continue
                path = job_dir(folder, job)
                for name in ("launch.json", "error.json"):
                    (path / name).unlink(missing_ok=True)
                path.mkdir(parents=True, exist_ok=True)
                log = (path / "worker.log").open("a")
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=device,
                    PYTHONPATH=str(frozen),
                    PYTHONDONTWRITEBYTECODE="1",
                    PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS=str(options.threads),
                    MKL_NUM_THREADS=str(options.threads),
                    OPENBLAS_NUM_THREADS=str(options.threads),
                    DGLBACKEND="pytorch",
                )
                command = [
                    sys.executable, "-B", "-u", "-m", "benchmark.reproduce_table9",
                    "job", "--run-dir", str(folder), "--job-id", job["id"],
                ]
                process = subprocess.Popen(
                    command, cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                nr.write_json(
                    path / "launch.json",
                    dict(pid=process.pid, device=device, utc=nr.utcnow(), command=command),
                )
                active[job["id"]] = dict(
                    process=process, job=job, device=device, log=log, started=time.monotonic()
                )
                used[device] += weight
                if job["method"] in ("geom", "sfgc"):
                    exclusive.add((job["dataset"], job["method"]))
                print(f"{nr.utcnow()} launched {job['id']} on {device}", flush=True)
            summary = export(folder, manifest)
            nr.write_json(
                folder / "heartbeat.json",
                dict(utc=nr.utcnow(), pid=os.getpid(), counts=summary["counts"],
                     active={k: v["device"] for k, v in active.items()}),
            )
            if not active and not pending:
                print(f"Queue finished: {summary['counts']}", flush=True)
                return
            time.sleep(15)


def published():
    rows = {}
    with (ROOT / "benchmark/results/gc4nc_table9.csv").open() as source:
        for row in csv.DictReader(source):
            rows[(row["Dataset"], row["Reduction rate (%)"])] = row
    return rows


def paper_value(paper, dataset, rate, method):
    if rate is None:
        label = LABELS[dataset][0]
    else:
        label = LABELS[dataset][RATES[dataset].index(rate)]
    value = paper[(CSV_NAMES[dataset], label)].get(NAMES[method], "")
    try:
        return float(value.split("±")[0])
    except ValueError:
        return None


def export(folder, manifest):
    paper = published()
    counts = Counter()
    cells = []
    groups = {}
    for job in manifest["jobs"]:
        counts[state(folder, job)] += 1
        groups.setdefault((job["dataset"], job["method"], job["rate"]), []).append(job)
    for (dataset, method, rate), jobs in groups.items():
        states = [state(folder, job) for job in jobs]
        cell = dict(dataset=dataset, method=method, rate=rate,
                    label=None if rate is None else LABELS[dataset][RATES[dataset].index(rate)],
                    paper=paper_value(paper, dataset, rate, method))
        if all(s == "complete" for s in states):
            records = [nr.read_json(job_dir(folder, job) / "result.json") for job in jobs]
            scores = [100 * e["test"] for r in records for e in r["evaluations"]]
            if len(scores) != EVAL_RUNS * len(jobs) or not all(map(math.isfinite, scores)):
                cell.update(status="invalid")
            else:
                cell.update(
                    status="complete",
                    mean=statistics.mean(scores),
                    std=statistics.pstdev(scores),
                    n=len(scores),
                    seconds=sum(r["elapsed_seconds"] for r in records),
                    reduced_nodes=[r.get("reduced_nodes") for r in records],
                )
        elif "failed" in states:
            kinds = {
                nr.read_json(job_dir(folder, job) / "error.json").get("kind", "ERROR")
                for job in jobs
                if (job_dir(folder, job) / "error.json").exists()
            }
            cell.update(status="failed", kind="OOM" if "OOM" in kinds else sorted(kinds)[0])
        else:
            cell.update(status="running" if "running" in states else "pending")
        cells.append(cell)
    summary = dict(updated_utc=nr.utcnow(), counts=dict(counts), cells=cells)
    nr.write_json(folder / "summary.json", summary)
    (folder / "results.md").write_text(render(cells, manifest, show_paper=True))
    return summary


def cell_text(cell):
    if cell is None:
        return "—"
    if cell["status"] == "complete":
        return f"{cell['mean']:.2f} ± {cell['std']:.2f}"
    if cell["status"] == "failed":
        return cell["kind"]
    return "…"


def render(cells, manifest, show_paper=False):
    index = {(c["dataset"], c["method"], c["rate"]): c for c in cells}
    lines = []
    for dataset in manifest["datasets"]:
        metric = "F1-macro" if dataset == "yelp" else "test accuracy"
        lines += [f"### {CSV_NAMES[dataset]} — {metric} (%)", ""]
        header = ["Family", "Method", *[f"{label}%" for label in LABELS[dataset]]]
        lines += ["| " + " | ".join(header) + " |", "| --- | --- | " + " | ".join(["---:"] * 3) + " |"]
        for method in [*manifest["methods"], "whole"]:
            values = []
            for rate in RATES[dataset]:
                cell = index.get((dataset, method, None if method == "whole" else rate))
                text = cell_text(cell)
                if show_paper and cell and cell.get("paper") is not None:
                    text += f" ({cell['paper']:.2f})"
                values.append(text)
            family = "Baseline" if method == "whole" else FAMILY.get(method, "Condensation")
            lines.append("| " + " | ".join([family, NAMES[method], *values]) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "job", "status"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(RATES))
    parser.add_argument("--methods", nargs="+", choices=tuple(NAMES))
    parser.add_argument("--devices", nargs="+", default=[], help="CUDA device or MIG UUIDs")
    parser.add_argument("--capacity", type=int, default=3, help="summed dataset weight per device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=12 * 3600)
    parser.add_argument("--job-id")
    options = parser.parse_args(argv)
    options.run_dir = options.run_dir.resolve()
    if options.action == "job":
        execute(options.run_dir, options.job_id)
        return
    manifest = initialize(options)
    if options.action == "run":
        if not options.devices:
            parser.error("run requires --devices")
        manage(options, manifest)
    else:
        print(json.dumps(export(options.run_dir, manifest)["counts"], indent=2))


if __name__ == "__main__":
    main()

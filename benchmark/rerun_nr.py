"""Resume NR experiments on idle, explicitly selected local GPUs.

Published results are immutable. Completed local cells are exported separately;
partial cells never receive a numeric result. Each seed owns its checkpoint path.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

from benchmark.reproduce import RATES

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("clustering", "doscondx", "gcdm", "simgc", "gecc")
SEEDS = (1, 2, 3)
DATA_SEED = 1
EVAL_RUNS = 10
START = "<!-- local-nr-results:start -->"
END = "<!-- local-nr-results:end -->"
NAMES = {
    "clustering": "Clustering",
    "doscondx": "DosCondX",
    "gcdm": "GCDM",
    "simgc": "SimGC",
    "gecc": "GECC",
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as temporary:
        temporary.write(serialized)
    Path(temporary.name).replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock_file(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def source_files():
    return sorted(
        [
            path
            for path in (ROOT / "graphslim").rglob("*")
            if path.suffix in (".py", ".json", ".npy", ".sh")
        ]
        + [
            ROOT / "benchmark" / "rerun_nr.py",
            ROOT / "benchmark" / "reproduce.py",
            ROOT / "pyproject.toml",
        ]
    )


def source_digest(files=None):
    digest = hashlib.sha256()
    for path in files or source_files():
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def plan_jobs(datasets=None, methods=None):
    jobs = []
    # Exercise every implementation early; later work covers the remaining rates.
    combinations = [
        (dataset, rate, method)
        for dataset in datasets or RATES
        for rate in RATES[dataset]
        for method in methods or METHODS
    ]
    for dataset, rate, method in combinations:
        for seed in SEEDS:
            reason = None
            if (
                method == "gecc"
                and not (ROOT / "graphslim" / "configs" / method / f"{dataset}.json").is_file()
            ):
                reason = f"No upstream GECC configuration for {dataset}; requires an explicit hyperparameter choice."
            jobs.append(
                {
                    "id": f"{dataset}__{method}__r{rate}__s{seed}",
                    "dataset": dataset,
                    "rate": rate,
                    "method": method,
                    "seed": seed,
                    "blocked_reason": reason,
                }
            )
    return sorted(
        jobs,
        key=lambda job: (
            job["dataset"] != "cora",
            list((datasets or RATES)).index(job["dataset"]),
            RATES[job["dataset"]].index(job["rate"]),
            job["seed"],
            (methods or METHODS).index(job["method"]),
        ),
    )


def initialize(run_dir, datasets=None, methods=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    files = source_files()
    manifest = {
        "created_utc": utcnow(),
        "source_sha256": source_digest(files),
        "upstream_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "protocol": {
            "data_seed": DATA_SEED,
            "reduction_seeds": list(SEEDS),
            "evaluation_seeds": "reduction_seed + [0..9]",
            "run_eval": EVAL_RUNS,
            "eval_epochs": 300,
            "run_inter_eval": 3,
            "eval_model": "GCN",
            "hidden": 256,
            "nlayers": 2,
            "uncertainty": "population standard deviation of all 30 test scores",
            "checkpoint_selection": "validation only",
            "rates": "fractions of labeled training nodes",
            "datasets": "GraphSlim loaders: Planetoid, OGB/GraphSAINT, Flickr, Reddit2, DGL FraudYelp; fixed data seed",
        },
        "jobs": plan_jobs(datasets, methods),
    }
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    write_json(manifest_path, manifest)
    return manifest


def job_path(run_dir, job):
    return run_dir / "jobs" / job["id"]


def job_state(run_dir, job):
    folder = job_path(run_dir, job)
    if job["blocked_reason"]:
        return "blocked"
    if (folder / "result.json").exists():
        return "complete"
    if (folder / "error.json").exists():
        return "failed"
    if (folder / "launch.json").exists():
        launch = read_json(folder / "launch.json")
        command = Path(f"/proc/{launch['pid']}/cmdline")
        try:
            if (
                job["id"].encode() in command.read_bytes()
                and b"benchmark.rerun_nr" in command.read_bytes()
            ):
                return "running"
        except (FileNotFoundError, ProcessLookupError):
            pass
        return "interrupted"
    return "pending"


def aggregate_cell(run_dir, jobs, expected_source):
    states = [job_state(run_dir, job) for job in jobs]
    if states == ["complete"] * len(SEEDS):
        records = [read_json(job_path(run_dir, job) / "result.json") for job in jobs]
        if any(record["source_sha256"] != expected_source for record in records):
            return {"status": "source mismatch", "value": "INVALID"}
        if len({json.dumps(record["data"], sort_keys=True) for record in records}) != 1:
            return {"status": "dataset mismatch", "value": "INVALID"}
        scores = [entry["test"] for record in records for entry in record["evaluations"]]
        if len(scores) != len(SEEDS) * EVAL_RUNS or not all(
            math.isfinite(score) and 0 <= score <= 1 for score in scores
        ):
            return {"status": "invalid scores", "value": "INVALID"}
        return {
            "status": "complete",
            "value": f"{100 * statistics.mean(scores):.2f} ± {100 * statistics.pstdev(scores):.2f}",
            "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores),
            "n": len(scores),
        }
    done = states.count("complete")
    if "blocked" in states:
        return {"status": "blocked", "value": "CONFIG"}
    if "failed" in states:
        return {"status": "failed", "value": f"FAILED ({done}/3)"}
    if "interrupted" in states:
        return {"status": "interrupted", "value": f"INTERRUPTED ({done}/3)"}
    return {
        "status": "running" if "running" in states else "pending",
        "value": f"PENDING ({done}/3)",
    }


def export(run_dir, *, update_readme=False):
    manifest = read_json(run_dir / "manifest.json")
    counts = Counter(job_state(run_dir, job) for job in manifest["jobs"])
    cells = []
    for dataset in dict.fromkeys(job["dataset"] for job in manifest["jobs"]):
        for rate in RATES[dataset]:
            for method in dict.fromkeys(job["method"] for job in manifest["jobs"]):
                selected = [
                    job
                    for job in manifest["jobs"]
                    if (job["dataset"], job["rate"], job["method"]) == (dataset, rate, method)
                ]
                if not selected:
                    continue
                cells.append(
                    {
                        "dataset": dataset,
                        "rate": rate,
                        "method": method,
                        **aggregate_cell(run_dir, selected, manifest["source_sha256"]),
                    }
                )
    write_json(
        run_dir / "summary.json",
        {"updated_utc": utcnow(), "run_dir": str(run_dir), "counts": dict(counts), "cells": cells},
    )
    lines = [
        "### Local NR rerun (separate from published Table 9)",
        "",
        f"Progress: {counts['complete']}/{len(manifest['jobs'])} reduction runs complete; {counts['failed']} failed; {counts['blocked']} require configuration.",
        "",
        "Each numeric cell requires all 3 reduction seeds × 10 downstream GCN fits, 300 evaluation epochs, 256 hidden units and 2 layers. Values are mean ± population standard deviation over the 30 test scores, in percent (Yelp: F1-macro). Synthetic checkpoints and GCN weights are selected on validation performance only. No incomplete cell is averaged or ranked.",
        "",
        "CLI rates below are fractions of labeled training nodes, not the whole-graph percentages printed in the published tables. Dataset splits are fixed with data seed 1; reduction seeds are 1, 2, 3. Method epochs and hyperparameters come from this repository's JSON configurations. GECC uses its implemented static mode (no evolving splits); its missing Yelp configuration is `CONFIG`, not a score.",
        "",
        "These are local reruns of the repository, not an exact replay of the published experiment: the Reddit loader uses the sparser Reddit2 variant; FraudYelp receives the repository's seeded classwise 80/10/10 split. See per-run dataset hashes/counts and resolved configurations for the exact inputs.",
        "",
        "| Dataset | CLI rate | Clustering | DosCondX | GCDM | SimGC | GECC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in dict.fromkeys(cell["dataset"] for cell in cells):
        for rate in RATES[dataset]:
            row = {
                cell["method"]: cell["value"]
                for cell in cells
                if cell["dataset"] == dataset and cell["rate"] == rate
            }
            lines.append(
                "| "
                + " | ".join([dataset, rate, *[row.get(method, "—") for method in METHODS]])
                + " |"
            )
    lines += [
        "",
        "`PENDING (n/3)` retains partial reductions without publishing a partial score. `FAILED`/`INTERRUPTED` retains an error record; `CONFIG` requires method configuration. Machine-readable scores, configurations, dataset hashes, source snapshot, checkpoint hashes, and logs are in the run directory.",
        "",
    ]
    rendered = "\n".join(lines)
    (run_dir / "results.md").write_text(rendered)
    if update_readme:
        readme_path = ROOT / "README.md"
        readme = readme_path.read_text()
        if readme.count(START) != 1 or readme.count(END) != 1:
            raise ValueError("README local result markers are missing or duplicated")
        before, rest = readme.split(START)
        _, after = rest.split(END)
        readme_path.write_text(before + START + "\n\n" + rendered + "\n" + END + after)
    return dict(counts)


def gpu_inventory():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    )
    busy = set(apps.splitlines())
    inventory = []
    for line in output.splitlines():
        index, uuid, memory, utilization = [item.strip() for item in line.split(",")]
        inventory.append(
            {
                "index": int(index),
                "uuid": uuid,
                "idle": uuid not in busy and int(memory) < 200 and int(utilization) < 5,
            }
        )
    return inventory


def acquire_gpu(gpu):
    handles = []
    try:
        # Honor the existing host's physical-index lock as well as a stable UUID lock.
        for path in (
            f"/tmp/graphslim-{gpu['uuid']}.lock",
            f"/tmp/ekpt-physical-gpu-{gpu['index']}.lock",
        ):
            handle = open(path, "a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        time.sleep(1)
        if not any(item["uuid"] == gpu["uuid"] and item["idle"] for item in gpu_inventory()):
            raise BlockingIOError("GPU is no longer idle")
        return handles
    except BlockingIOError:
        for handle in handles:
            handle.close()
        return None


def run_queue(options, run_dir):
    manifest = initialize(run_dir, options.datasets, options.methods)
    if source_digest() != manifest["source_sha256"]:
        raise RuntimeError("Source differs from the frozen manifest. Use a new run directory.")
    active = {}
    allowed = set(options.gpus)
    with lock_file(run_dir / "manager.lock", blocking=False):
        write_json(
            run_dir / "manager.json",
            {
                "pid": os.getpid(),
                "gpus_authorized": sorted(allowed),
                "started_utc": utcnow(),
                "python": sys.executable,
            },
        )
        try:
            while True:
                for uuid, (process, handles, log, job) in list(active.items()):
                    if process.poll() is None:
                        continue
                    if (
                        not (job_path(run_dir, job) / "result.json").exists()
                        and not (job_path(run_dir, job) / "error.json").exists()
                    ):
                        write_json(
                            job_path(run_dir, job) / "error.json",
                            {
                                "error": "Child exited without a result",
                                "exit_code": process.returncode,
                                "utc": utcnow(),
                            },
                        )
                    log.close()
                    for handle in handles:
                        handle.close()
                    del active[uuid]
                    print(f"{utcnow()} finished {job['id']} exit={process.returncode}", flush=True)
                pending = [
                    job
                    for job in manifest["jobs"]
                    if job_state(run_dir, job) in ("pending", "interrupted")
                ]
                if pending:
                    for gpu in gpu_inventory():
                        if (
                            gpu["index"] not in allowed
                            or not gpu["idle"]
                            or gpu["uuid"] in active
                            or not pending
                        ):
                            continue
                        handles = acquire_gpu(gpu)
                        if not handles:
                            continue
                        job = pending.pop(0)
                        folder = job_path(run_dir, job)
                        folder.mkdir(parents=True, exist_ok=True)
                        log = (folder / "console.log").open("a")
                        env = {
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                            "OMP_NUM_THREADS": str(options.threads),
                            "MKL_NUM_THREADS": str(options.threads),
                            "OPENBLAS_NUM_THREADS": str(options.threads),
                            "PYTHONUNBUFFERED": "1",
                            "DGLBACKEND": "pytorch",
                        }
                        command = [
                            sys.executable,
                            "-m",
                            "benchmark.rerun_nr",
                            "job",
                            "--run-dir",
                            str(run_dir),
                            "--job-id",
                            job["id"],
                        ]
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            pass_fds=tuple(handle.fileno() for handle in handles),
                            start_new_session=True,
                        )
                        write_json(
                            folder / "launch.json",
                            {
                                "pid": process.pid,
                                "gpu": gpu,
                                "started_utc": utcnow(),
                                "command": command,
                            },
                        )
                        active[gpu["uuid"]] = process, handles, log, job
                        print(
                            f"{utcnow()} launched {job['id']} physical GPU {gpu['index']} pid={process.pid}",
                            flush=True,
                        )
                counts = export(run_dir, update_readme=options.update_readme)
                write_json(
                    run_dir / "heartbeat.json",
                    {
                        "utc": utcnow(),
                        "pid": os.getpid(),
                        "counts": counts,
                        "active": {uuid: job["id"] for uuid, (_, _, _, job) in active.items()},
                    },
                )
                if (
                    not active
                    and not counts.get("pending", 0)
                    and not counts.get("running", 0)
                    and not counts.get("interrupted", 0)
                ):
                    print(f"Queue finished: {counts}", flush=True)
                    return
                time.sleep(10)
        finally:
            # Child processes inherit the locks, so interruption cannot release an
            # executing GPU lease prematurely. A resumed manager discovers them.
            for _, handles, log, _ in active.values():
                log.close()
                for handle in handles:
                    handle.close()


def serializable(value):
    if callable(value):
        return value.__name__
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value
    return str(value)


def array_hash(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(memoryview(value.copy(order="C")).cast("B"))
    return digest.hexdigest()


def execute_job(run_dir, job_id):
    manifest = read_json(run_dir / "manifest.json")
    job = next(job for job in manifest["jobs"] if job["id"] == job_id)
    folder = job_path(run_dir, job)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError(
                "Scientific source changed after planning; refusing a mixed-version result."
            )
        import torch
        from graphslim.config import get_args
        from graphslim.dataset import get_dataset
        from graphslim.train_all import run
        from graphslim.utils import seed_everything

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each job requires exactly one UUID-masked CUDA GPU.")
        args = get_args(
            [
                "-D",
                job["dataset"],
                "-M",
                job["method"],
                "-R",
                job["rate"],
                "--seed",
                str(job["seed"]),
                "-G",
                "0",
                "--run_reduction",
                "1",
                "--run_eval",
                str(EVAL_RUNS),
                "--run_inter_eval",
                "3",
                "--eval_epochs",
                "300",
                "--eval_model",
                "GCN",
                "--hidden",
                "256",
                "--nlayers",
                "2",
                "--load_path",
                str(ROOT / "data"),
                "--save_path",
                str(folder / "checkpoints"),
            ]
        )
        write_json(
            folder / "config.json", {key: serializable(value) for key, value in vars(args).items()}
        )
        environment = {
            "python": sys.version,
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cuda": torch.version.cuda,
            "packages": {
                dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            },
        }
        write_json(folder / "environment.json", environment)
        data_record = {}

        def loader(name, config, path):
            with lock_file(ROOT / "data" / f".{name}.prepare.lock"):
                seed_everything(DATA_SEED)
                graph = get_dataset(name, config, path)
            data_record.update(
                {
                    "nodes": graph.num_nodes,
                    "classes": int(graph.nclass),
                    "feature_shape": list(graph.feat_full.shape),
                    "adj_nnz": int(graph.adj_full.nnz),
                    "split_sizes": {
                        split: int(getattr(graph, f"idx_{split}").numel())
                        for split in ("train", "val", "test")
                    },
                    "hashes": {
                        split: array_hash(getattr(graph, f"idx_{split}"))
                        for split in ("train", "val", "test")
                    },
                    "labels_sha256": array_hash(graph.labels_full),
                    "features_sha256": array_hash(graph.feat_full),
                    "adj_sha256": {
                        part: array_hash(getattr(graph.adj_full, part))
                        for part in ("indptr", "indices", "data")
                    },
                }
            )
            write_json(folder / "dataset.json", data_record)
            return graph

        def record(config, graph, evaluator, result):
            prefix = folder / "checkpoints" / "reduced_graph" / job["method"]
            suffix = f"{job['dataset']}_{config.reduction_rate}_{job['seed']}.pt"
            paths = [prefix / f"{kind}_{suffix}" for kind in ("adj", "feat", "label")]
            if not all(path.is_file() for path in paths):
                raise RuntimeError("Missing method checkpoint; no fallback graph may be scored.")
            if len(evaluator.last_results) != EVAL_RUNS:
                raise RuntimeError("Incomplete downstream evaluation")
            if not torch.isfinite(graph.feat_syn).all() or not torch.isfinite(graph.adj_syn).all():
                raise RuntimeError("Non-finite synthetic graph")
            write_json(
                folder / "result.json",
                {
                    "job": job,
                    "completed_utc": utcnow(),
                    "elapsed_seconds": time.monotonic() - started,
                    "source_sha256": manifest["source_sha256"],
                    "data": data_record,
                    "evaluations": evaluator.last_results,
                    "mean": float(result[0]),
                    "std": float(result[1]),
                    "synthetic_nodes": int(graph.feat_syn.shape[0]),
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                    "checkpoint_sha256": {
                        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths
                    },
                    "resolved_config": {
                        key: serializable(value) for key, value in vars(config).items()
                    },
                },
            )

        run(args, load_dataset=loader, on_result=record)
    except Exception as exc:
        write_json(
            folder / "error.json",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "utc": utcnow(),
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "job", "status", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(RATES))
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=list(range(6)),
        help="authorized physical indices; only idle GPUs are acquired",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--job-id")
    parser.add_argument("--update-readme", action="store_true")
    options = parser.parse_args(argv)
    run_dir = options.run_dir.resolve()
    if options.action == "run":
        run_queue(options, run_dir)
    elif options.action == "job":
        execute_job(run_dir, options.job_id)
    else:
        if options.action == "plan":
            initialize(run_dir, options.datasets, options.methods)
        print(json.dumps(export(run_dir, update_readme=options.update_readme), indent=2))


if __name__ == "__main__":
    main()

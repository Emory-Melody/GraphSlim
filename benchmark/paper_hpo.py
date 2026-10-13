"""Paper-sourced GECC/SimGC tuning with validation-only selection and GPU leases.

This is a local extension, not an exact paper replay (Yelp/Pubmed, Reddit2,
three reduction seeds, and the canonical reduction used to search are explicit).
Published and earlier untuned results are never overwritten.
"""

import argparse
from collections import Counter
from itertools import product, zip_longest
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

from benchmark.reproduce import RATES
from benchmark import rerun_nr as nr

ROOT = nr.ROOT
DATASETS = ("citeseer", "cora", "pubmed", "ogbn-arxiv", "flickr", "reddit", "yelp")
EVAL_SEEDS = tuple(range(10))
START = "<!-- paper-hpo-results:start -->"
END = "<!-- paper-hpo-results:end -->"
SOURCES = {
    "gecc": "https://arxiv.org/html/2502.17614#S5.SS1",
    "gc4nc": "https://arxiv.org/html/2406.16715v3#A1.SS8.SSS2",
    "simgc": "https://arxiv.org/html/2403.14951#S5.SS1",
    "simgc_code": "https://github.com/BangHonor/SimGC/tree/1b3d1d59d9afaf2c9c0c7b652fb183db2a5ffa01",
}


def gcn_grid(dataset=None):
    keys = ("hidden", "lr", "nlayers", "weight_decay", "dropout")
    candidates = [
        dict(zip(keys, values))
        for values in product(
            (64, 128, 256), (0.01, 0.001), (2, 3, 4), (0, 0.0005, 0.001), (0, 0.5)
        )
    ]
    if dataset in ("cora", "citeseer"):
        # Keep the released citation baseline reachable, outside Table 11's widths.
        candidates.append(
            {"hidden": 1024, "lr": 0.001, "nlayers": 2, "weight_decay": 0, "dropout": 0}
        )
    return candidates


def dataset_gcn_grid(manifest, dataset):
    """Old frozen manifests keep their original 108-candidate roster."""
    return manifest.get("gcn_grids", {}).get(dataset, manifest.get("gcn_grid", gcn_grid()))


def gecc_grid():
    values = tuple(round(index / 10, 1) for index in range(-3, 10))
    # alpha_0 -> agg_gamma, alpha_1 -> agg_alpha, alpha_2 -> agg_beta.
    return [
        dict(zip(("agg_gamma", "agg_alpha", "agg_beta"), weights))
        for weights in product(values, repeat=3)
    ]


def group(job):
    return f"{job['method']}__{job['dataset']}"


def reduced_id(dataset, rate, seed):
    return f"reduce__simgc__{dataset}__r{rate}__s{seed}"


def roster(datasets=None, methods=None):
    reductions = [
        {
            "id": reduced_id(dataset, rate, seed),
            "kind": "reduce",
            "method": "simgc",
            "dataset": dataset,
            "rate": rate,
            "seed": seed,
            "depends": [],
        }
        for dataset in DATASETS
        for rate in reversed(RATES[dataset])
        for seed in nr.SEEDS
    ]
    reductions.sort(key=lambda job: (job["seed"] != 1, job["rate"] != RATES[job["dataset"]][-1]))
    gcn_jobs = [
        {
            "id": f"gcn__simgc__{dataset}__b{block:02}",
            "kind": "gcn",
            "method": "simgc",
            "dataset": dataset,
            "rate": RATES[dataset][-1],
            "seed": 1,
            "candidates": list(range(block * 12, min((block + 1) * 12, len(gcn_grid(dataset))))),
            "depends": [reduced_id(dataset, RATES[dataset][-1], 1)],
        }
        for dataset in DATASETS
        for block in range((len(gcn_grid(dataset)) + 11) // 12)
    ]
    gecc_jobs = [
        {
            "id": f"gecc__yelp__b{block:03}",
            "kind": "gecc",
            "method": "gecc",
            "dataset": "yelp",
            "rate": RATES["yelp"][-1],
            "seed": 1,
            "candidates": list(range(block * 13, (block + 1) * 13)),
            "depends": [],
        }
        for block in range(169)
    ]
    finals = [
        {
            "id": f"final__{method}__{dataset}__r{rate}__s{seed}",
            "kind": "final",
            "method": method,
            "dataset": dataset,
            "rate": rate,
            "seed": seed,
            "depends": [reduced_id(dataset, rate, seed)] if method == "simgc" else [],
        }
        for method, datasets in (("simgc", DATASETS), ("gecc", ("yelp",)))
        for dataset in datasets
        for rate in RATES[dataset]
        for seed in nr.SEEDS
    ]
    canonical = [
        job for job in reductions if job["seed"] == 1 and job["rate"] == RATES[job["dataset"]][-1]
    ]
    remaining = [job for job in reductions if job not in canonical]
    tasks = [job for batch in zip_longest(gecc_jobs, gcn_jobs, remaining) for job in batch if job]
    selected_datasets = set(DATASETS if datasets is None else datasets)
    selected_methods = set(("simgc", "gecc") if methods is None else methods)
    if not selected_datasets or not selected_datasets <= set(DATASETS):
        raise ValueError("Select supported nonempty datasets")
    if not selected_methods or not selected_methods <= {"simgc", "gecc"}:
        raise ValueError("Select supported nonempty methods")
    jobs = [
        job
        for job in finals + canonical + tasks
        if job["dataset"] in selected_datasets and job["method"] in selected_methods
    ]
    if not jobs:
        raise ValueError("The selected methods have no jobs on these datasets")
    return jobs


def scientific_files():
    return nr.source_files() + [
        Path(__file__).resolve(),
        ROOT / "benchmark/results/paper_hpo_provenance.json",
        ROOT / "benchmark/package_results.py",
        ROOT / "benchmark/results/package_runs.json",
    ]


def initialize(folder, datasets=None, methods=None, readme_section=None):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        manifest = nr.read_json(folder / "manifest.json")
        for key, requested in (("datasets", datasets), ("methods", methods)):
            if requested is not None:
                existing = {job[key[:-1]] for job in manifest["jobs"]}
                if existing != set(requested):
                    raise RuntimeError("Requested scope differs from the frozen manifest")
        if (
            readme_section is not None
            and manifest.get("readme_section", "paper-hpo") != readme_section
        ):
            raise RuntimeError("README destination differs from the frozen manifest")
        return manifest
    jobs = roster(datasets, methods)
    scoped_datasets = [
        dataset for dataset in DATASETS if any(job["dataset"] == dataset for job in jobs)
    ]
    files = scientific_files()
    manifest = {
        "created_utc": nr.utcnow(),
        "source_sha256": nr.source_digest(files),
        "sources": SOURCES,
        "gcn_grid": gcn_grid(),
        "gcn_grids": {dataset: gcn_grid(dataset) for dataset in scoped_datasets},
        "readme_section": readme_section or "paper-hpo",
        "gecc_grid": gecc_grid(),
        "protocol": {
            "selection": "validation mean only; deterministic lowest candidate ID breaks ties",
            "test_policy": "no test metric computed during reduction/search; unseal after all candidates for a group complete",
            "search_reduction_seed": 1,
            "search_rate": "highest CLI rate per dataset",
            "evaluation_seeds": list(EVAL_SEEDS),
            "eval_epochs": 300,
            "intermediate_evaluations": 10,
            "run_inter_eval": 3,
            "final_reduction_seeds": list(nr.SEEDS),
            "data_seed": nr.DATA_SEED,
            "gecc": {
                "depth": 2,
                "fuzziness": 1.0,
                "rep_fuzz": 1,
                "balance_alpha": 1,
                "weight_decay": [0.0005, 0.001],
                "hidden": 256,
                "nlayers": 2,
                "lr": 0.01,
                "dropout": 0.5,
                "mode": "static hard k-means++",
            },
            "simgc": "fixed released author settings; GC4NC Table 11's 108 downstream candidates plus the released GCN baseline for Cora/Citeseer (109 each); no invented condensation grid",
            "simgc_citation": {
                "pre_norm": False,
                "activation": None,
                "validation_hidden": 2048,
                "final_default_hidden": 1024,
                "validation_epochs": 3000,
                "downstream_epochs": 3001,
                "val_interval": 100,
                "condensation_validation_interval": 100,
                "run_inter_eval": 1,
                "note": "Author loops: intermediate range(3000); final range(3000+1). Final HPO may select a different width/depth/dropout, not an exact paper replay.",
            },
            "extensions": [
                "GECC paper has no Yelp experiment: apply its large-dataset grid to local FraudYelp",
                "SimGC paper has no Pubmed/Yelp settings: use common transductive/inductive author defaults, explicitly not published optima",
                "Reddit loader uses Reddit2; FraudYelp uses fixed seed-1 classwise 80/10/10 split",
                "canonical seed-1 reduction for search; selected config reused for all rates and 3 final reductions",
                "final cells pool 3 reductions x 10 GCN fits; population standard deviation, not standard error",
            ],
        },
        "jobs": jobs,
    }
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    nr.write_json(folder / "manifest.json", manifest)
    return manifest


def job_path(folder, job):
    return folder / "jobs" / job["id"]


def state(folder, job):
    path = job_path(folder, job)
    if (path / "result.json").exists():
        return "complete"
    if (path / "error.json").exists():
        return "failed"
    if (path / "launch.json").exists():
        launch = nr.read_json(path / "launch.json")
        try:
            command = Path(f"/proc/{launch['pid']}/cmdline").read_bytes()
            if job["id"].encode() in command and b"benchmark.paper_hpo" in command:
                return "running"
        except (FileNotFoundError, ProcessLookupError):
            pass
    return "pending"


def ready(folder, job, states):
    return all(states[dependency] == "complete" for dependency in job["depends"]) and (
        job["kind"] != "final" or (folder / "selected" / f"{group(job)}.json").is_file()
    )


def select_complete_groups(folder, manifest, states):
    groups = {}
    for job in manifest["jobs"]:
        if job["kind"] in ("gcn", "gecc"):
            groups.setdefault(group(job), []).append(job)
    for name, jobs in groups.items():
        target = folder / "selected" / f"{name}.json"
        if target.exists() or not all(states[job["id"]] == "complete" for job in jobs):
            continue
        records = [nr.read_json(job_path(folder, job) / "result.json") for job in jobs]
        if any(record["source_sha256"] != manifest["source_sha256"] for record in records):
            raise RuntimeError("Cannot select mixed scientific source versions")
        if len({json.dumps(record["data"], sort_keys=True) for record in records}) != 1:
            raise RuntimeError("Cannot select mixed dataset versions")
        candidates = [candidate for record in records for candidate in record["candidates"]]
        expected = (
            len(dataset_gcn_grid(manifest, jobs[0]["dataset"]))
            if jobs[0]["kind"] == "gcn"
            else 4394
        )
        if len(candidates) != expected or {candidate["id"] for candidate in candidates} != set(
            range(expected)
        ):
            raise RuntimeError("Incomplete or duplicate search roster")
        for candidate in candidates:
            evaluations = candidate["evaluations"]
            if len(evaluations) != 10 or {entry["seed"] for entry in evaluations} != set(
                EVAL_SEEDS
            ):
                raise RuntimeError("Incomplete candidate evaluation seeds")
            if any(set(entry) != {"seed", "validation"} for entry in evaluations):
                raise RuntimeError("Search records must not contain test metrics")
            scores = [entry["validation"] for entry in evaluations]
            if (
                not all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
                or statistics.mean(scores) != candidate["validation_mean"]
            ):
                raise RuntimeError("Invalid raw validation metrics or aggregate")
        winner = max(
            candidates, key=lambda candidate: (candidate["validation_mean"], -candidate["id"])
        )
        nr.write_json(
            target,
            {
                "frozen_utc": nr.utcnow(),
                "source_sha256": manifest["source_sha256"],
                "data": records[0]["data"],
                "candidate_count": expected,
                "winner": winner,
                "search_results_sha256": {
                    job["id"]: hashlib.sha256(
                        (job_path(folder, job) / "result.json").read_bytes()
                    ).hexdigest()
                    for job in jobs
                },
            },
        )


def export(folder, manifest, update_readme=False):
    states = {job["id"]: state(folder, job) for job in manifest["jobs"]}
    select_complete_groups(folder, manifest, states)
    counts = Counter(states.values())
    candidate_counts = {
        kind: sum(
            len(list((job_path(folder, job) / "candidates").glob("*.json")))
            for job in manifest["jobs"]
            if job["kind"] == kind
        )
        for kind in ("gcn", "gecc")
    }
    by_kind = {
        kind: dict(Counter(states[job["id"]] for job in manifest["jobs"] if job["kind"] == kind))
        for kind in ("reduce", "gcn", "gecc", "final")
    }
    cells = []
    for method, datasets in (("simgc", DATASETS), ("gecc", ("yelp",))):
        for dataset in datasets:
            for rate in RATES[dataset]:
                jobs = [
                    job
                    for job in manifest["jobs"]
                    if job["kind"] == "final"
                    and (job["method"], job["dataset"], job["rate"]) == (method, dataset, rate)
                ]
                if not jobs:
                    continue
                if len(jobs) != len(nr.SEEDS) or {job["seed"] for job in jobs} != set(nr.SEEDS):
                    raise RuntimeError("Incomplete final reduction roster")
                cell = {"method": method, "dataset": dataset, "rate": rate, "value": "PENDING"}
                if all(states[job["id"]] == "complete" for job in jobs):
                    records = [nr.read_json(job_path(folder, job) / "result.json") for job in jobs]
                    for record in records:
                        evaluations = record["evaluations"]
                        if len(evaluations) != len(EVAL_SEEDS) or {
                            entry["seed"] for entry in evaluations
                        } != set(EVAL_SEEDS):
                            raise RuntimeError("Incomplete or duplicate final evaluation seeds")
                    if (
                        len({record["selection_sha256"] for record in records}) != 1
                        or any(
                            record["source_sha256"] != manifest["source_sha256"]
                            for record in records
                        )
                        or len({json.dumps(record["data"], sort_keys=True) for record in records})
                        != 1
                    ):
                        raise RuntimeError("Mixed selection, source or split in final cell")
                    scores = [
                        evaluation["test"]
                        for record in records
                        for evaluation in record["evaluations"]
                    ]
                    if len(scores) != 30 or not all(
                        math.isfinite(score) and 0 <= score <= 1 for score in scores
                    ):
                        raise RuntimeError("Invalid final scores")
                    cell.update(mean=statistics.mean(scores), std=statistics.pstdev(scores), n=30)
                    cell["value"] = f"{cell['mean'] * 100:.2f} ± {cell['std'] * 100:.2f}"
                cells.append(cell)
    nr.write_json(
        folder / "summary.json",
        {
            "utc": nr.utcnow(),
            "counts": dict(counts),
            "by_kind": by_kind,
            "search_candidate_counts": candidate_counts,
            "cells": cells,
        },
    )
    gcn_datasets = {job["dataset"] for job in manifest["jobs"] if job["kind"] == "gcn"}
    gcn_total = sum(len(dataset_gcn_grid(manifest, dataset)) for dataset in gcn_datasets)
    gecc_total = sum(
        len(job["candidates"]) * 2 for job in manifest["jobs"] if job["kind"] == "gecc"
    )
    repaired = manifest.get("readme_section") == "repaired-hpo"
    lines = [
        "### Repaired package HPO (fresh source; historical batches retained)"
        if repaired
        else "### Paper-sourced HPO (new local batch; earlier untuned results retained)",
        "",
        f"Jobs: {counts['complete']}/{len(manifest['jobs'])} complete; {counts['running']} running; {counts['failed']} failed.",
        "",
        f"Completed validation candidates: GECC {candidate_counts['gecc']}/{gecc_total}; SimGC downstream GCN {candidate_counts['gcn']}/{gcn_total}.",
        "",
        "GECC–Yelp: exhaustive 4,394 settings (13³ propagation weights × 2 weight decays), K=2, hard k-means++, one restart. SimGC: corrected author implementation/settings, then the 108-config downstream GCN grid from GC4NC Table 11. These are different search spaces, not interchangeable.",
        "",
        "Searches use only validation metrics, ten fixed evaluation seeds, 300 epochs, and the seed-1 graph at the highest CLI reduction rate. A group must finish every candidate before its choice is frozen. That choice is reused across all three rates; each final cell requires 3 reduction seeds × 10 GCN fits. Values are test mean ± population standard deviation (%) (Yelp: F1-macro).",
        "",
        "Scope caveats: GECC has no published Yelp optimum; SimGC has no published Pubmed/Yelp optimum (common author defaults are extensions). Reddit uses Reddit2; FraudYelp uses the fixed local seed-1 split. SimGC synthetic alignment gradients, teacher propagation order/normalization, and smoothness sign were repaired against the pinned author code. GECC inductive propagation depth and downstream training-mode dropout were repaired. Do not interpret this batch as a hyperparameter-only comparison or an exact paper replay.",
        "",
        f"Sources: [GECC §5.1]({SOURCES['gecc']}), [GC4NC Table 11]({SOURCES['gc4nc']}), [SimGC §5.1]({SOURCES['simgc']}), [pinned SimGC code]({SOURCES['simgc_code']}).",
        "",
        "| Dataset | CLI rate | SimGC (author settings + GCN HPO) | GECC (paper grid, Yelp extension) |",
        "| --- | ---: | ---: | ---: |",
    ]
    if "gcn_grids" in manifest:
        lines[6] = (
            (
                "GECC–Yelp: exhaustive 4,394 paper-grid settings. "
                if gecc_total
                else "No new GECC search in this batch; completed GECC results remain in their historical source batch. "
            )
            + "SimGC: 108 GC4NC downstream "
            "GCN candidates per dataset plus the released baseline for Cora/Citeseer (109 each)."
        )
        lines[8] = (
            "Validation-only search with ten fixed evaluation seeds; Cora/Citeseer use "
            "the released no-activation GCN profile (3,001 downstream updates, validation "
            "every 100); others use the 300-epoch local protocol. Condensation selection "
            "is separate (citation: hidden 2048, 3,000 updates). Full groups freeze before "
            "testing. Final cells pool 3 reduction seeds × 10 GCN fits, test mean ± "
            "population standard deviation (%; Yelp: F1-macro)."
        )
    if repaired:
        lines[10] += (
            " Cora/Citeseer now use the released effective raw features, separate "
            "citation model profiles and Glorot initialization; SimGC uses per-class "
            "ceiling budgets. These method-specific repairs change the scientific "
            "protocol; do not rank across source batches as one controlled experiment."
        )
    for dataset in DATASETS:
        for rate in RATES[dataset]:
            row = {
                cell["method"]: cell["value"]
                for cell in cells
                if (cell["dataset"], cell["rate"]) == (dataset, rate)
            }
            if not row:
                continue
            lines.append(
                f"| {dataset} | {rate} | {row.get('simgc', '—')} | {row.get('gecc', '—')} |"
            )
    lines += [
        "",
        f"Artifacts: `{folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder}` (manifest, frozen source, raw validation scores, selections, test records, checkpoint hashes and logs). `PENDING` is not a numeric result.",
        "",
    ]
    rendered = "\n".join(lines)
    (folder / "results.md").write_text(rendered)
    if update_readme:
        path = ROOT / "README.md"
        text = path.read_text()
        start = "<!-- repaired-hpo-results:start -->" if repaired else START
        end = "<!-- repaired-hpo-results:end -->" if repaired else END
        if text.count(start) != 1 or text.count(end) != 1:
            raise RuntimeError("README HPO result markers missing/duplicated")
        before, rest = text.split(start)
        _, after = rest.split(end)
        path.write_text(before + start + "\n\n" + rendered + "\n" + end + after)
        if repaired:
            from benchmark import package_results

            package_results.export(update_readme=True)
    return states, dict(counts)


def run_queue(folder, options):
    manifest = initialize(folder, options.datasets, options.methods, options.readme_section)
    if nr.source_digest(scientific_files()) != manifest["source_sha256"]:
        raise RuntimeError("Scientific source changed; create a new batch")
    active = {}
    with nr.lock_file(folder / "manager.lock", blocking=False):
        nr.write_json(folder / "manager.json", {"pid": os.getpid(), "started_utc": nr.utcnow()})
        try:
            while True:
                for uuid, (process, handles, log, job) in list(active.items()):
                    if process.poll() is None:
                        continue
                    path = job_path(folder, job)
                    if not (path / "result.json").exists() and not (path / "error.json").exists():
                        nr.write_json(
                            path / "error.json",
                            {"error": f"Child exited {process.returncode}", "utc": nr.utcnow()},
                        )
                    log.close()
                    for handle in handles:
                        handle.close()
                    del active[uuid]
                    print(
                        f"{nr.utcnow()} finished {job['id']} exit={process.returncode}", flush=True
                    )
                states, counts = export(folder, manifest, options.update_readme)
                pending = [
                    job
                    for job in manifest["jobs"]
                    if states[job["id"]] == "pending" and ready(folder, job, states)
                ]
                for gpu in nr.gpu_inventory():
                    if (
                        not pending
                        or gpu["index"] not in options.gpus
                        or not gpu["idle"]
                        or gpu["uuid"] in active
                    ):
                        continue
                    handles = nr.acquire_gpu(gpu)
                    if not handles:
                        continue
                    job = pending.pop(0)
                    path = job_path(folder, job)
                    path.mkdir(parents=True, exist_ok=True)
                    log = (path / "console.log").open("a")
                    env = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                        "DGLBACKEND": "pytorch",
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": str(options.threads),
                        "MKL_NUM_THREADS": str(options.threads),
                        "OPENBLAS_NUM_THREADS": str(options.threads),
                    }
                    command = [
                        sys.executable,
                        "-m",
                        "benchmark.paper_hpo",
                        "job",
                        "--run-dir",
                        str(folder),
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
                    nr.write_json(
                        path / "launch.json",
                        {"pid": process.pid, "gpu": gpu, "command": command, "utc": nr.utcnow()},
                    )
                    active[gpu["uuid"]] = process, handles, log, job
                    print(
                        f"{nr.utcnow()} launched {job['id']} physical GPU {gpu['index']} pid={process.pid}",
                        flush=True,
                    )
                nr.write_json(
                    folder / "heartbeat.json",
                    {
                        "utc": nr.utcnow(),
                        "counts": counts,
                        "pid": os.getpid(),
                        "active": {uuid: job["id"] for uuid, (_, _, _, job) in active.items()},
                    },
                )
                if (
                    not active
                    and not any(states[job["id"]] == "running" for job in manifest["jobs"])
                    and not pending
                ):
                    print(
                        f"Queue terminal: {counts}; unresolved dependencies are not scored.",
                        flush=True,
                    )
                    return
                time.sleep(10)
        finally:
            for _, handles, log, _ in active.values():
                log.close()
                for handle in handles:
                    handle.close()


def config_for(folder, job):
    from graphslim.config import get_args

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
            "--load_path",
            str(ROOT / "data"),
            "--save_path",
            str(folder / "checkpoints"),
        ]
    )
    args.validation_only, args.eval_mode = True, "cross"
    if job["method"] != "simgc" or job["dataset"] not in ("cora", "citeseer"):
        args.lr, args.hidden, args.nlayers = 0.01, 256, 2
        args.dropout, args.weight_decay = 0.5, 0.0005
        args.eval_epochs, args.run_inter_eval = 300, 3
    args.checkpoints = list(range(args.eval_interval, args.epochs + 1, args.eval_interval))
    if job["method"] == "gecc":
        args.depth, args.fuzziness, args.rep_fuzz, args.balance_alpha = 2, 1.0, 1, 1
    return args


def dataset_record(graph):
    return {
        "nodes": graph.num_nodes,
        "classes": int(graph.nclass),
        "feature_shape": list(graph.feat_full.shape),
        "adj_nnz": int(graph.adj_full.nnz),
        "split_sizes": {
            split: int(getattr(graph, f"idx_{split}").numel()) for split in ("train", "val", "test")
        },
        "hashes": {
            split: nr.array_hash(getattr(graph, f"idx_{split}"))
            for split in ("train", "val", "test")
        },
        "labels_sha256": nr.array_hash(graph.labels_full),
        "features_sha256": nr.array_hash(graph.feat_full),
        "adj_sha256": {
            part: nr.array_hash(getattr(graph.adj_full, part))
            for part in ("indptr", "indices", "data")
        },
    }


def checkpoint_paths(args):
    prefix = Path(args.save_path) / "reduced_graph" / args.method
    suffix = f"{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
    paths = [prefix / f"{kind}_{suffix}" for kind in ("adj", "feat", "label")]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("Missing reduced graph; refusing fallback")
    return paths


def load_checkpoint(graph, args):
    import torch

    graph.adj_syn, graph.feat_syn, graph.labels_syn = [
        torch.load(path, map_location=args.device, weights_only=True)
        for path in checkpoint_paths(args)
    ]
    if not torch.isfinite(graph.feat_syn).all() or not torch.isfinite(graph.adj_syn).all():
        raise RuntimeError("Non-finite reduced graph")


def use_reduction(folder, job, args, data, source):
    """Verify the canonical/replicate graph before using it in a later stage."""
    parent = folder / "jobs" / reduced_id(job["dataset"], job["rate"], job["seed"])
    record = nr.read_json(parent / "result.json")
    if record["source_sha256"] != source or record["data"] != data:
        raise RuntimeError("Reduced graph source/split mismatch")
    args.save_path = str(parent / "checkpoints")
    hashes = {
        str(item.relative_to(folder)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in checkpoint_paths(args)
    }
    if hashes != record["checkpoint_sha256"]:
        raise RuntimeError("Reduced graph checkpoint hash mismatch")


def evaluate(graph, args, *, test=False):
    from graphslim.models import GCN
    from graphslim.utils import seed_everything

    evaluations = []
    for seed in EVAL_SEEDS:
        seed_everything(seed)
        model = GCN(graph.feat_syn.shape[1], args.hidden, graph.nclass, args, mode="cross").to(
            args.device
        )
        validation = model.fit_with_val(
            graph, train_iters=args.eval_epochs, normadj=True, reduced=True, setting=args.setting
        )
        entry = {"seed": seed, "validation": validation}
        if test:
            entry["test"] = model.test(graph, setting=args.setting)
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for key, value in entry.items()
            if key != "seed"
        ):
            raise RuntimeError("Non-finite or invalid evaluation metric")
        evaluations.append(entry)
    return evaluations


def apply_params(args, params):
    for key, value in params.items():
        setattr(args, key, value)
    # Legacy soft-label GECC fits consult eval_wd; keep it aligned with the grid.
    args.eval_wd = args.weight_decay


def candidate_record(candidate_id, params, evaluations):
    return {
        "id": candidate_id,
        "params": params,
        "evaluations": evaluations,
        "validation_mean": statistics.mean(entry["validation"] for entry in evaluations),
    }


def execute_job(folder, job_id):
    manifest = nr.read_json(folder / "manifest.json")
    job = next(job for job in manifest["jobs"] if job["id"] == job_id)
    path = job_path(folder, job)
    path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if nr.source_digest(scientific_files()) != manifest["source_sha256"]:
            raise RuntimeError("Scientific source changed after planning")
        import torch
        from graphslim.dataset import get_dataset
        from graphslim.reduction import create_reducer
        from graphslim.utils import seed_everything

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("A job needs exactly one leased, UUID-masked GPU")
        args = config_for(path, job)
        with nr.lock_file(ROOT / "data" / f".{job['dataset']}.prepare.lock"):
            seed_everything(nr.DATA_SEED)
            graph = get_dataset(job["dataset"], args, args.load_path)
        data = dataset_record(graph)
        nr.write_json(path / "dataset.json", data)
        nr.write_json(
            path / "config.json", {key: nr.serializable(value) for key, value in vars(args).items()}
        )
        nr.write_json(
            path / "environment.json",
            {
                "python": sys.version,
                "cuda": torch.version.cuda,
                "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
                "gpu_name": torch.cuda.get_device_name(0),
                "packages": {
                    dist.metadata["Name"]: dist.version
                    for dist in importlib.metadata.distributions()
                },
            },
        )
        result = {"job": job, "source_sha256": manifest["source_sha256"], "data": data}
        if job["kind"] == "reduce":
            seed_everything(job["seed"])
            create_reducer("simgc", setting=args.setting, data=graph, args=args).reduce(
                graph, verbose=False
            )
            load_checkpoint(graph, args)
            result["evaluations"] = evaluate(graph, args)
        elif job["kind"] in ("gcn", "gecc"):
            candidates = []
            if job["kind"] == "gcn":
                use_reduction(folder, job, args, data, manifest["source_sha256"])
                load_checkpoint(graph, args)
            for index in job["candidates"]:
                for wd_index in range(2) if job["kind"] == "gecc" else range(1):
                    candidate_id = index * 2 + wd_index if job["kind"] == "gecc" else index
                    candidate_path = path / "candidates" / f"{candidate_id:04}.json"
                    if candidate_path.exists():
                        candidates.append(nr.read_json(candidate_path))
                        continue
                    if job["kind"] == "gecc":
                        params = {
                            **manifest["gecc_grid"][index],
                            "weight_decay": (0.0005, 0.001)[wd_index],
                        }
                        args.save_path = str(path / "graphs" / f"{index:04}")
                        apply_params(args, params)
                        try:
                            load_checkpoint(graph, args)
                        except RuntimeError as exc:
                            if "Missing reduced graph" not in str(exc):
                                raise
                            graph.reset()
                            seed_everything(job["seed"])
                            create_reducer(
                                "gecc", setting=args.setting, data=graph, args=args
                            ).reduce(graph, verbose=False)
                            load_checkpoint(graph, args)
                    else:
                        params = dataset_gcn_grid(manifest, job["dataset"])[index]
                        apply_params(args, params)
                    record = candidate_record(candidate_id, params, evaluate(graph, args))
                    record["checkpoint_sha256"] = {
                        str(item.relative_to(folder)): hashlib.sha256(item.read_bytes()).hexdigest()
                        for item in checkpoint_paths(args)
                    }
                    nr.write_json(candidate_path, record)
                    candidates.append(record)
                    print(
                        f"{nr.utcnow()} candidate {candidate_id} validation={record['validation_mean']:.6f}",
                        flush=True,
                    )
            result["candidates"] = candidates
        else:
            selection_path = folder / "selected" / f"{group(job)}.json"
            selection = nr.read_json(selection_path)
            if selection["source_sha256"] != manifest["source_sha256"] or selection["data"] != data:
                raise RuntimeError("Final evaluation source/split differs from selection")
            apply_params(args, selection["winner"]["params"])
            if job["method"] == "simgc":
                use_reduction(folder, job, args, data, manifest["source_sha256"])
            else:
                seed_everything(job["seed"])
                create_reducer("gecc", setting=args.setting, data=graph, args=args).reduce(
                    graph, verbose=False
                )
            load_checkpoint(graph, args)
            result["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            result["evaluations"] = evaluate(graph, args, test=True)
        if job["kind"] not in ("gcn", "gecc"):
            result["checkpoint_sha256"] = {
                str(item.relative_to(folder)): hashlib.sha256(item.read_bytes()).hexdigest()
                for item in checkpoint_paths(args)
            }
        result.update(
            completed_utc=nr.utcnow(),
            elapsed_seconds=time.monotonic() - started,
            resolved_config={key: nr.serializable(value) for key, value in vars(args).items()},
        )
        nr.write_json(path / "result.json", result)
    except Exception as exc:
        nr.write_json(
            path / "error.json",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "utc": nr.utcnow(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "job", "status"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=("simgc", "gecc"))
    parser.add_argument("--readme-section", choices=("paper-hpo", "repaired-hpo"))
    options = parser.parse_args(argv)
    folder = options.run_dir.resolve()
    if options.action == "run":
        run_queue(folder, options)
    elif options.action == "job":
        execute_job(folder, options.job_id)
    else:
        manifest = (
            initialize(folder, options.datasets, options.methods, options.readme_section)
            if options.action == "plan"
            else nr.read_json(folder / "manifest.json")
        )
        _, counts = export(folder, manifest, options.update_readme)
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

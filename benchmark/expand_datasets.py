"""Frozen, resumable three-stage dataset expansion with explicit resource failures.

Stage 1: original seven datasets. Stage 2: two official heterophilous graphs.
Stage 3: ogbn-products with official OGB splits. Numerical cells require every
declared reduction/evaluation seed. Different historical batches are never pooled.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import fcntl
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
METHODS = ("cgc", "clustgdd", "gcgp", "nope", "random")
OPTIONAL_METHODS = ("gecc", "clustering", "vng")
STAGES = (
    ("cora", "citeseer", "pubmed", "ogbn-arxiv", "flickr", "reddit", "yelp"),
    ("roman-empire", "amazon-ratings"),
    ("ogbn-products",),
)
EXPANSION_RATES = {
    **RATES,
    "roman-empire": ("0.005", "0.01", "0.02"),
    "amazon-ratings": ("0.005", "0.01", "0.02"),
    "ogbn-products": ("0.0005", "0.001", "0.002"),
}


def plan(datasets=None, methods=METHODS, seeds=(1, 2, 3)):
    selected = set(datasets or sum(STAGES, ()))
    if selected - set(sum(STAGES, ())):
        raise ValueError("Unknown dataset")
    jobs = []
    for stage, roster in enumerate(STAGES, 1):
        for dataset in roster:
            if dataset not in selected:
                continue
            jobs.append(
                dict(
                    id=f"whole__{dataset}",
                    stage=stage,
                    dataset=dataset,
                    method="whole",
                    kind="whole",
                    seed=1,
                    rate=None,
                )
            )
            for rate in EXPANSION_RATES[dataset]:
                for method in methods:
                    for seed in seeds:
                        jobs.append(
                            dict(
                                id=f"{dataset}__{method}__r{rate}__s{seed}",
                                stage=stage,
                                dataset=dataset,
                                method=method,
                                kind="reduction",
                                seed=seed,
                                rate=rate,
                            )
                        )
            if "gcgp" in methods:
                jobs.append(
                    dict(
                        id=f"parity__{dataset}",
                        stage=stage,
                        dataset=dataset,
                        method="gcgp",
                        kind="parity",
                        seed=1,
                        rate=EXPANSION_RATES[dataset][-1],
                    )
                )
    return jobs


def science_files():
    return sorted(
        set(
            nr.source_files()
            + [
                ROOT / "benchmark/expand_datasets.py",
                ROOT / "benchmark/gcgp_parity.py",
                ROOT / "benchmark/verify_lightweight.py",
                ROOT / "benchmark/paper_hpo.py",
            ]
        )
    )


def initialize(options):
    folder = options.run_dir
    folder.mkdir(parents=True, exist_ok=True)
    jobs = plan(options.datasets, options.methods, options.seeds)
    if getattr(options, "reductions_only", False):
        jobs = [job for job in jobs if job["kind"] == "reduction"]
    protocol = dict(
        reduction_seeds=options.seeds,
        evaluation_seeds=options.eval_seeds,
        downstream="GCN",
        hidden=256,
        layers=2,
        eval_epochs=options.eval_epochs,
        learning_rate=0.01,
        weight_decay=0.0005,
        dropout=0.0,
        activation="relu",
        val_interval=options.val_interval,
        data_seed=1,
        dataset_split_id=0,
        threads=options.threads,
        selection="validation only",
        reduction="package method defaults; no HPO; eager backend",
        rate_denominator="labeled training nodes; NOPE reports actual retained nodes",
        parity="author SGNK eager versus native CUDA kernel capture; GP/Adam eager",
        timeout_seconds=options.timeout,
        max_rss_gib=options.max_rss_gib,
        failure_policy="terminal failures retained; no partial numeric cells",
        precision="float32; CGC reconstruction solve float64 as in package",
        metric="accuracy except FraudYelp macro-F1",
        settings="original package trans/ind settings; new datasets transductive",
        comparison_scope="new source/budgets only; no pooling of historical batches",
    )
    if "gecc" in options.methods:
        protocol["implementation_notes"] = {
            "gecc": "Full normalized sparse propagation on Products replaces the legacy sampled overwrite branch; bounded exact squared-distance arithmetic; sparse diagnostic assignments use original node IDs.",
            "cluster_ops": "Native sklearn algorithms/restarts retained; segmented and sparse aggregation can change floating-point summation order. See operator validation; no historical source pooling.",
        }
    if (folder / "manifest.json").exists():
        manifest = nr.read_json(folder / "manifest.json")
        if manifest["jobs"] != jobs or manifest["protocol"] != protocol:
            raise RuntimeError("Frozen roster/protocol differs; choose a new run directory")
        return manifest
    files = science_files()
    manifest = dict(
        schema_version=1,
        created_utc=nr.utcnow(),
        jobs=jobs,
        protocol=protocol,
        gpus=options.gpus,
        stage_order=[list(stage) for stage in STAGES],
        environment=dict(python=sys.version, executable=sys.executable),
    )
    previous_package = {}
    if options.inherit_run:
        previous_manifest = nr.read_json(options.inherit_run / "manifest.json")
        previous_zip = options.inherit_run / "source.zip"
        if (
            hashlib.sha256(previous_zip.read_bytes()).hexdigest()
            != previous_manifest["archive_sha256"]
        ):
            raise RuntimeError("Previous source archive hash mismatch")
        with zipfile.ZipFile(previous_zip) as archive:
            previous_package = {
                name: archive.read(name)
                for name in archive.namelist()
                if name.startswith("graphslim/") and name != "graphslim/dataset/loader.py"
            }
    # Keep the already-measured package implementation frozen even if independent
    # development changes the live checkout while this experiment is running.
    source_hash = hashlib.sha256()
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = str(path.relative_to(ROOT))
            content = previous_package.get(name, path.read_bytes())
            source_hash.update(name.encode())
            source_hash.update(content)
            archive.writestr(name, content)
    manifest["source_sha256"] = source_hash.hexdigest()
    manifest["archive_sha256"] = hashlib.sha256((folder / "source.zip").read_bytes()).hexdigest()
    frozen = folder / "frozen_workspace"
    frozen.mkdir(exist_ok=True)
    with zipfile.ZipFile(folder / "source.zip") as archive:
        archive.extractall(frozen)
    for name in ("data", "runs"):
        (frozen / name).symlink_to(ROOT / name, target_is_directory=True)
    if options.inherit_run:
        inherit_complete_cells(options.inherit_run, folder, manifest)
    nr.write_json(folder / "manifest.json", manifest)
    export(folder, manifest)
    return manifest


def inherit_complete_cells(previous_folder, folder, manifest):
    """Reuse only complete, homogeneous cells with an explicit source audit."""
    previous_folder = previous_folder.resolve()
    previous = nr.read_json(previous_folder / "manifest.json")
    if previous["protocol"] != manifest["protocol"] or previous["jobs"] != manifest["jobs"]:
        raise RuntimeError("Cannot inherit a different protocol/roster")
    archive_path = previous_folder / "source.zip"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != previous["archive_sha256"]:
        raise RuntimeError("Previous source archive hash mismatch")
    # The allowed change is PyG cache deserialization in the new OGB branch.
    # Every other package file must still match the sealed original source.
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.startswith("graphslim/") and name != "graphslim/dataset/loader.py":
                candidate = folder / "frozen_workspace" / name
                if not candidate.exists() or archive.read(name) != candidate.read_bytes():
                    raise RuntimeError(f"Package arithmetic changed outside OGB loader: {name}")
    summary = export(previous_folder, previous)
    complete_groups = {
        (cell["stage"], cell["dataset"], cell["method"], cell["rate"])
        for cell in summary["cells"]
        if cell["status"] == "complete"
    }
    inherited = {}
    for job in manifest["jobs"]:
        complete = (job["stage"], job["dataset"], job["method"], job["rate"]) in complete_groups
        if job["kind"] == "parity":
            outcome = terminal(previous_folder, job)
            complete = bool(outcome and outcome["status"] == "complete")
        if not complete:
            continue
        target = folder / "jobs" / job["id"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(previous_folder / "jobs" / job["id"], target_is_directory=True)
        inherited[job["id"]] = job_source(previous, job)
    manifest["inherited_sources"] = inherited
    manifest["inheritance"] = dict(
        run_dir=str(previous_folder),
        source_sha256=previous["source_sha256"],
        archive_sha256=previous["archive_sha256"],
        reused_jobs=len(inherited),
        audit="All package science files except dataset/loader.py match previous sealed archive byte for byte; only complete seed groups reused",
        reason="Explicit PyG safe-global context for OGB processed-cache compatibility; primary original-dataset math/settings unchanged",
    )


def terminal(folder, job):
    return (
        nr.read_json(folder / "jobs" / job["id"] / "terminal.json")
        if (folder / "jobs" / job["id"] / "terminal.json").exists()
        else None
    )


def job_source(manifest, job):
    return manifest.get("inherited_sources", {}).get(job["id"], manifest["source_sha256"])


def export(folder, manifest):
    """Publish a cell only with exact source, data, seed and evaluation coverage."""
    outcomes, records = {}, {}
    for job in manifest["jobs"]:
        outcome = terminal(folder, job)
        if outcome:
            if outcome["job"] != job or outcome["source_sha256"] != job_source(manifest, job):
                raise RuntimeError("Mixed terminal source/roster")
            outcomes[job["id"]] = outcome["status"]
            if outcome["status"] == "complete":
                record = nr.read_json(folder / "jobs" / job["id"] / "result.json")
                if record["source_sha256"] != job_source(manifest, job) or record["job"] != job:
                    raise RuntimeError("Mixed result source/roster")
                records[job["id"]] = record
    cells, raw = [], []
    groups = {}
    for job in manifest["jobs"]:
        if job["kind"] != "parity":
            groups.setdefault(
                (job["stage"], job["dataset"], job["method"], job["rate"]), []
            ).append(job)
    for (stage, dataset, method, rate), jobs in groups.items():
        cell = dict(
            stage=stage,
            dataset=dataset,
            method=method,
            rate=rate,
            expected_reductions=len(jobs),
            complete_reductions=0,
            status="pending",
        )
        available = [records[job["id"]] for job in jobs if job["id"] in records]
        cell["complete_reductions"] = len(available)
        states = [outcomes.get(job["id"], "pending") for job in jobs]
        if len(available) == len(jobs):
            data_versions = {json.dumps(record["data"], sort_keys=True) for record in available}
            if len(data_versions) != 1:
                raise RuntimeError("Mixed dataset versions in one cell")
            source_versions = {record["source_sha256"] for record in available}
            if len(source_versions) != 1:
                raise RuntimeError("Mixed source versions in one cell")
            values = []
            for record in available:
                evaluations = record["evaluations"]
                if sorted(value["seed"] for value in evaluations) != sorted(
                    manifest["protocol"]["evaluation_seeds"]
                ):
                    raise RuntimeError("Missing/duplicate evaluation seed")
                for value in evaluations:
                    if not math.isfinite(value["test"]) or not 0 <= value["test"] <= 1:
                        raise RuntimeError("Invalid score")
                    raw.append(
                        dict(
                            dataset=dataset,
                            method=method,
                            rate=rate,
                            source_sha256=record["source_sha256"],
                            reduction_seed=record["job"]["seed"],
                            **value,
                        )
                    )
                values.extend(value["test"] for value in evaluations)
            cell.update(
                status="complete",
                n=len(values),
                mean=statistics.mean(values),
                std=statistics.pstdev(values),
                source_sha256=available[0]["source_sha256"],
                synthetic_nodes=[record["reduction"]["synthetic_nodes"] for record in available],
                reduction_seconds=[record["timing"]["reduction_seconds"] for record in available],
                max_gpu_allocated_bytes=max(
                    record["resources"]["gpu_peak_allocated_bytes"] for record in available
                ),
                max_rss_bytes=max(record["resources"]["max_rss_bytes"] for record in available),
            )
        elif all(value != "pending" for value in states):
            cell["status"] = "unavailable"
            cell["reasons"] = states
        cells.append(cell)
    counts = Counter(outcomes.values())
    stage_status = []
    for stage in range(1, 4):
        roster = [job for job in manifest["jobs"] if job["stage"] == stage]
        states = Counter(outcomes.get(job["id"], "pending") for job in roster)
        stage_status.append(
            dict(
                stage=stage,
                expected=len(roster),
                counts=dict(states),
                terminal=states["pending"] == 0,
            )
        )
    summary = dict(
        updated_utc=nr.utcnow(),
        source_sha256=manifest["source_sha256"],
        expected_jobs=len(manifest["jobs"]),
        terminal_jobs=len(outcomes),
        counts=dict(counts),
        stages=stage_status,
        cells=cells,
        numeric_cells=sum(cell["status"] == "complete" for cell in cells),
        expected_cells=len(cells),
        parity={
            job["dataset"]: records.get(job["id"], {}).get(
                "parity", outcomes.get(job["id"], "pending")
            )
            for job in manifest["jobs"]
            if job["kind"] == "parity"
        },
    )
    nr.write_json(folder / "summary.json", summary)
    if raw:
        with (folder / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
            writer.writeheader()
            writer.writerows(raw)
    lines = [
        "# Ordered dataset expansion",
        "",
        f"Updated: {summary['updated_utc']}",
        "",
        f"Terminal jobs: {len(outcomes)}/{len(manifest['jobs'])}; complete numeric cells: {summary['numeric_cells']}/{len(cells)}.",
        "",
        "New frozen-source experiments; method-specific package defaults, fixed GCN, validation-only model selection. Historical scores are not pooled. Standard deviations are population SD over the declared GCN fits, not confidence intervals.",
        "",
        "| Stage | Dataset | Method | Training-relative rate | Test mean ± SD (%) | Fits | Actual synthetic nodes | Median reduction seconds | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        complete = cell["status"] == "complete"
        score = (
            f"{100 * cell['mean']:.2f} ± {100 * cell['std']:.2f}"
            if complete
            else cell["status"].upper()
        )
        nodes = "/".join(map(str, cell["synthetic_nodes"])) if complete else "—"
        seconds = f"{statistics.median(cell['reduction_seconds']):.3f}" if complete else "—"
        source = cell.get("source_sha256", manifest["source_sha256"])[:12]
        lines.append(
            f"| {cell['stage']} | {cell['dataset']} | {cell['method']} | {cell['rate'] or 'whole'} | {score} | {cell.get('n', '—')} | {nodes} | {seconds} | {source} |"
        )
    if manifest.get("inheritance"):
        lines += [
            "",
            f"Complete cells inherited from `{manifest['inheritance']['run_dir']}` retain their original source hash. Only whole complete seed groups are reused; incomplete groups are not spliced. The archived package mathematics outside the OGB loader are identical. See manifest inheritance audit.",
        ]
    if manifest.get("continuation"):
        continuation = manifest["continuation"]
        lines += [
            "",
            f"Evaluation-only continuation of `{continuation['parent']}`: {len(continuation['jobs'])} timed-out jobs reuse hash-verified reductions and completed fits. The original terminal records remain archived. Frozen computation, data and seeds are unchanged; the new per-attempt wall limit is {manifest['protocol']['timeout_seconds']} seconds. Runtime totals retain saved successful work but exclude interrupted attempts; see the continuation manifest for original outcomes and limits.",
        ]
    lines += [
        "",
        "NOPE is the existing training-budget condensation adapter, not the original-node classifier from its paper. Unreachable targets, unlabeled-supernode filtering and class loss remain disclosed in per-job metadata. Timeout/resource failures are unavailable, never zero. GPU peaks are allocator peaks; per-job RSS is process peak, including data and training. Three stages execute in order after all prior-stage jobs are terminal, including retained failures.",
        "",
    ]
    (folder / "results.md").write_text("\n".join(lines))
    return summary


def resource_usage():
    import resource
    import torch

    return dict(
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )


def worker(options, manifest, job):
    import torch
    from benchmark.paper_hpo import dataset_record, checkpoint_paths
    from graphslim.config import get_args
    from graphslim.dataset import get_dataset
    from graphslim.evaluation import Evaluator
    from graphslim.reduction import create_reducer
    from graphslim.utils import seed_everything

    folder = options.run_dir / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    wall_started = time.monotonic()
    if nr.source_digest(science_files()) != manifest["source_sha256"]:
        raise RuntimeError("Frozen source digest mismatch")
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    requested_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if requested_gpu.startswith("GPU-") and (
        gpu_uuid.removeprefix("GPU-") != requested_gpu.removeprefix("GPU-")
    ):
        raise RuntimeError("Visible CUDA device differs from the locked GPU UUID")
    protocol = manifest["protocol"]
    torch.set_num_threads(protocol["threads"])
    args = get_args(
        [
            "-D",
            job["dataset"],
            "-M",
            job["method"] if job["method"] != "whole" else "random",
            "-R",
            job["rate"] or EXPANSION_RATES[job["dataset"]][-1],
            "-G",
            "0",
            "-S",
            str(job["seed"]),
            "--run_eval",
            "1",
            "--hidden",
            "256",
            "--eval_epochs",
            str(protocol["eval_epochs"]),
            "--save_path",
            str(folder / "checkpoints"),
            "--dataset_split_id",
            str(protocol["dataset_split_id"]),
        ]
    )
    args.val_interval = protocol["val_interval"]
    args.eval_mode = "cross"
    args.lr = 0.01
    args.weight_decay = 0.0005
    args.dropout = 0.0
    args.activation = "relu"
    args.nlayers = 2
    args.with_bn = False
    args.cuda_graph = False
    args.eval_wd = 0.0005

    def phase(name):
        nr.write_json(folder / "heartbeat.json", dict(pid=os.getpid(), phase=name, utc=nr.utcnow()))
        print(f"{nr.utcnow()} {job['id']} {name}", flush=True)

    phase("loading")
    loading_started = time.monotonic()
    seed_everything(protocol["data_seed"])
    if job["dataset"] == "ogbn-products":
        # Dataset download is part of the requested third stage; never wait for
        # the OGB library's interactive size confirmation in a detached worker.
        import ogb.nodeproppred.dataset_pyg as ogb_pyg

        ogb_pyg.decide_download = lambda url: True
    with nr.lock_file(ROOT / "data" / f".{job['dataset']}.prepare.lock"):
        graph = get_dataset(args.dataset, args, str(ROOT / "data"))
    data = dataset_record(graph)
    data["split_provenance"] = getattr(
        graph, "split_provenance", dict(source="package", data_seed=1)
    )
    load_seconds = time.monotonic() - loading_started
    nr.write_json(folder / "dataset.json", data)
    nr.write_json(
        folder / "config.json", {key: nr.serializable(value) for key, value in vars(args).items()}
    )
    torch.cuda.reset_peak_memory_stats()
    record = dict(
        job=job,
        source_sha256=manifest["source_sha256"],
        data=data,
        protocol=protocol,
        gpu_name=torch.cuda.get_device_name(0),
        gpu_uuid=gpu_uuid,
        evaluations=[],
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda),
        reduction=dict(synthetic_nodes=graph.num_nodes, uses_test_labels=False),
    )
    saved_resources = []
    if job["kind"] == "parity":
        phase("parity")
        from benchmark.gcgp_parity import check

        record["parity"] = check(graph, args, protocol["reduction_seeds"], folder)
        if not record["parity"]["passed"]:
            nr.write_json(folder / "result.json", record)
            raise RuntimeError("GCGP numerical parity failed; see result.json")
    else:
        phase("reduction")
        before = time.monotonic()
        seed_everything(job["seed"])
        if job["kind"] == "reduction":
            saved_reduction = folder / "reduction.json"
            if saved_reduction.exists():
                saved = nr.read_json(saved_reduction)
                if (
                    saved["source_sha256"] != manifest["source_sha256"]
                    or saved["data"] != data
                    or saved["job"] != job
                ):
                    raise RuntimeError("Saved reduction source/data/roster mismatch")
                record["reduction"] = saved["reduction"]
                record["checkpoint_sha256"] = saved["checkpoint_sha256"]
                reduction_seconds = saved["seconds"]
                saved_resources.append(saved["resources"])
            else:
                graph = create_reducer(job["method"], args.setting, graph, args).reduce(
                    graph, verbose=False
                )
                torch.cuda.synchronize()
                record["reduction"] = getattr(
                    graph,
                    "reduction_metadata",
                    dict(synthetic_nodes=len(graph.feat_syn), uses_test_labels=False),
                )
                reduction_seconds = time.monotonic() - before
                record["checkpoint_sha256"] = {
                    str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in checkpoint_paths(args)
                }
                nr.write_json(
                    saved_reduction,
                    dict(
                        source_sha256=manifest["source_sha256"],
                        data=data,
                        job=job,
                        reduction=record["reduction"],
                        checkpoint_sha256=record["checkpoint_sha256"],
                        seconds=reduction_seconds,
                        resources=resource_usage(),
                    ),
                )
            paths = checkpoint_paths(args)
            hashes = {
                str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
            if hashes != record["checkpoint_sha256"]:
                raise RuntimeError("Saved reduction checkpoint hash mismatch")
            load_checkpoint(graph, args)
            hard_labels = (
                graph.labels_syn.argmax(1)
                if graph.labels_syn.ndim == 2
                else graph.labels_syn.long()
            )
            histogram = torch.bincount(hard_labels, minlength=graph.nclass).cpu().tolist()
            record["reduction"].update(
                training_nodes=len(graph.idx_train),
                requested_training_rate=args.reduction_rate,
                argmax_class_histogram=histogram,
                missing_argmax_classes=[
                    index for index, count in enumerate(histogram) if count == 0
                ],
                label_type="continuous optimized targets"
                if graph.labels_syn.ndim == 2
                else "hard classes",
            )
        else:
            reduction_seconds = 0.0
        record["timing"] = dict(
            reduction_seconds=reduction_seconds, loading_and_hash_seconds=load_seconds
        )
        for eval_seed in protocol["evaluation_seeds"]:
            saved = folder / "evaluations" / f"GCN__s{eval_seed}.json"
            if saved.exists():
                previous = nr.read_json(saved)
                if (
                    previous["source_sha256"] != manifest["source_sha256"]
                    or previous["job"] != job
                    or previous["data"] != data
                ):
                    raise RuntimeError("Saved evaluation source/data/roster mismatch")
                if previous.get("checkpoint_sha256") != record.get("checkpoint_sha256"):
                    raise RuntimeError("Saved evaluation checkpoint mismatch")
                if previous["score"]["seed"] != eval_seed:
                    raise RuntimeError("Saved evaluation seed mismatch")
                record["evaluations"].append(previous["score"])
                saved_resources.append(previous["resources"])
                continue
            seed_everything(eval_seed)
            model_args = deepcopy(args)
            model_args.seed = eval_seed
            phase(f"evaluation_{eval_seed}")
            before = time.monotonic()
            validation, test = Evaluator(model_args).test(
                graph, "GCN", verbose=False, reduced=job["kind"] == "reduction", mode="cross"
            )
            torch.cuda.synchronize()
            raw = dict(
                seed=eval_seed,
                validation=float(validation),
                test=float(test),
                seconds=time.monotonic() - before,
            )
            if not all(math.isfinite(raw[key]) for key in ("validation", "test")):
                raise FloatingPointError("Non-finite downstream score")
            record["evaluations"].append(raw)
            nr.write_json(
                saved,
                dict(
                    score=raw,
                    source_sha256=manifest["source_sha256"],
                    data=data,
                    job=job,
                    checkpoint_sha256=record.get("checkpoint_sha256"),
                    resources=resource_usage(),
                ),
            )
            torch.cuda.empty_cache()
        record["timing"]["total_compute_seconds"] = reduction_seconds + sum(
            raw["seconds"] for raw in record["evaluations"]
        )
        record["timing"]["worker_wall_seconds"] = time.monotonic() - wall_started
        current_resources = resource_usage()
        record["resources"] = {
            key: max([value[key] for value in saved_resources] + [current])
            for key, current in current_resources.items()
        }
    nr.write_json(folder / "result.json", record)
    nr.write_json(
        folder / "terminal.json",
        dict(job=job, source_sha256=manifest["source_sha256"], status="complete", utc=nr.utcnow()),
    )
    phase("complete")


def load_checkpoint(graph, args):
    """Strict reload for both dense synthetic graphs and sparse random coresets."""
    import torch
    from benchmark.paper_hpo import checkpoint_paths

    graph.adj_syn, graph.feat_syn, graph.labels_syn = [
        torch.load(path, map_location=args.device, weights_only=True)
        for path in checkpoint_paths(args)
    ]
    for tensor in (graph.adj_syn, graph.feat_syn, graph.labels_syn):
        values = (
            tensor if tensor.layout == torch.strided else tensor.to_sparse_coo().coalesce().values()
        )
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite checkpoint")


def manage(options, manifest):
    folder = options.run_dir
    frozen = folder / "frozen_workspace"
    active = {}
    with nr.lock_file(folder / "manager.lock", blocking=False):
        while True:
            unfinished = [job for job in manifest["jobs"] if terminal(folder, job) is None]
            if not unfinished and not active:
                break
            stage = min(job["stage"] for job in unfinished) if unfinished else None
            pending = [
                job
                for job in unfinished
                if job["stage"] == stage
                and job["id"] not in {item["job"]["id"] for item in active.values()}
            ]
            # A surviving worker keeps its job after a manager interruption.
            # Resume waits for it instead of launching a duplicate on another GPU.
            pending = [job for job in pending if not worker_alive(folder, job)]
            inventory = {item["index"]: item for item in nr.gpu_inventory()}
            for gpu in options.gpus:
                if not pending or gpu in active or not inventory.get(gpu, {}).get("idle", False):
                    continue
                handles = nr.acquire_gpu(inventory[gpu])
                if handles is None:
                    continue
                job = pending.pop(0)
                job_folder = folder / "jobs" / job["id"]
                job_folder.mkdir(parents=True, exist_ok=True)
                log = (job_folder / "worker.log").open("a")
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=inventory[gpu]["uuid"],
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    PYTHONPATH=str(frozen),
                    PYTHONDONTWRITEBYTECODE="1",
                    OMP_NUM_THREADS=str(options.threads),
                    MKL_NUM_THREADS=str(options.threads),
                    OPENBLAS_NUM_THREADS=str(options.threads),
                )
                command = [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "benchmark.expand_datasets",
                    "--run-dir",
                    str(folder),
                    "--worker",
                    job["id"],
                ]
                process = subprocess.Popen(
                    command,
                    cwd=frozen,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                nr.write_json(
                    job_folder / "launch.json",
                    dict(
                        pid=process.pid,
                        physical_gpu=gpu,
                        gpu_uuid=inventory[gpu]["uuid"],
                        utc=nr.utcnow(),
                        command=command,
                    ),
                )
                active[gpu] = dict(
                    process=process, job=job, started=time.monotonic(), log=log, handles=handles
                )
                print(f"{nr.utcnow()} GPU {gpu} started {job['id']}", flush=True)
            for gpu, item in list(active.items()):
                process = item["process"]
                timeout = time.monotonic() - item["started"] > options.timeout
                rss_path = Path(f"/proc/{process.pid}/status")
                rss = 0
                if rss_path.exists():
                    for line in rss_path.read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1]) * 1024
                resource_limit = rss > options.max_rss_gib * 1024**3
                if process.poll() is None and (timeout or resource_limit):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                if process.poll() is not None:
                    job = item["job"]
                    if terminal(folder, job) is None:
                        nr.write_json(
                            folder / "jobs" / job["id"] / "terminal.json",
                            dict(
                                job=job,
                                source_sha256=manifest["source_sha256"],
                                utc=nr.utcnow(),
                                status="timeout"
                                if timeout
                                else "resource_limit"
                                if resource_limit
                                else "failed",
                                exit_code=process.returncode,
                                elapsed_seconds=time.monotonic() - item["started"],
                                reason="See worker.log and heartbeat.json; no numeric partial cell",
                            ),
                        )
                    item["log"].close()
                    for handle in item["handles"]:
                        handle.close()
                    del active[gpu]
                    print(
                        f"{nr.utcnow()} finished {job['id']}: {terminal(folder, job)['status']}",
                        flush=True,
                    )
            summary = export(folder, manifest)
            nr.write_json(
                folder / "heartbeat.json",
                dict(
                    pid=os.getpid(),
                    utc=nr.utcnow(),
                    stage=stage,
                    terminal_jobs=summary["terminal_jobs"],
                    expected_jobs=summary["expected_jobs"],
                    active={
                        str(gpu): dict(
                            pid=item["process"].pid,
                            job=item["job"]["id"],
                            elapsed_seconds=time.monotonic() - item["started"],
                        )
                        for gpu, item in active.items()
                    },
                ),
            )
            time.sleep(5)
    summary = export(folder, manifest)
    nr.write_json(
        folder / "complete.json",
        dict(
            utc=nr.utcnow(),
            **{
                key: summary[key]
                for key in (
                    "source_sha256",
                    "counts",
                    "terminal_jobs",
                    "expected_jobs",
                    "numeric_cells",
                    "expected_cells",
                )
            },
        ),
    )
    print(json.dumps(summary["counts"]), flush=True)


def worker_alive(folder, job):
    lock = folder / "jobs" / job["id"] / "worker.lock"
    if lock.exists():
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                return True
    heartbeat = folder / "jobs" / job["id"] / "heartbeat.json"
    if not heartbeat.exists():
        return False
    record = nr.read_json(heartbeat)
    command = Path(f"/proc/{record['pid']}/cmdline")
    return (
        command.exists()
        and job["id"].encode() in command.read_bytes()
        and b"benchmark.expand_datasets" in command.read_bytes()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS + OPTIONAL_METHODS, default=list(METHODS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--gpus", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-epochs", type=int, default=300)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--max-rss-gib", type=float, default=160)
    parser.add_argument("--inherit-run", type=Path)
    parser.add_argument(
        "--reductions-only",
        action="store_true",
        help="Run only selected reductions; omit whole-graph and parity jobs",
    )
    parser.add_argument("--worker")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    options = parser.parse_args(argv)
    options.run_dir = options.run_dir.resolve()
    if options.worker or options.export:
        manifest = nr.read_json(options.run_dir / "manifest.json")
    else:
        if len(set(options.seeds)) != len(options.seeds) or len(set(options.eval_seeds)) != len(
            options.eval_seeds
        ):
            parser.error("Seeds must be distinct")
        manifest = initialize(options)
    if options.export:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in export(options.run_dir, manifest).items()
                    if key not in ("cells", "parity")
                },
                indent=2,
            )
        )
    elif options.worker:
        try:
            with nr.lock_file(
                options.run_dir / "jobs" / options.worker / "worker.lock", blocking=False
            ):
                worker(
                    options,
                    manifest,
                    next(job for job in manifest["jobs"] if job["id"] == options.worker),
                )
        except Exception:
            nr.write_json(
                options.run_dir / "jobs" / options.worker / "error.json",
                dict(error=traceback.format_exc(), utc=nr.utcnow()),
            )
            raise
    elif not options.initialize_only:
        manage(options, manifest)


if __name__ == "__main__":
    main()

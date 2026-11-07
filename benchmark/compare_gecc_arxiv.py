# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Add a matched GECC cohort to the frozen full-ogbn-arxiv comparison.

The author reducer is pinned independently. GraphSlim uses the author's uniform
centroid initialization in this comparison, so initializer changes are excluded
from the measured implementation difference. The ordinary package is untouched.
"""

import argparse
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "graphslim").is_dir())
REFERENCE = (
    "GECC",
    "d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f",
    "https://github.com/Emory-Melody/GECC",
)


def configure(folder):
    source = folder / "source"
    sys.path.insert(0, str(source if source.exists() else ROOT))
    from benchmark import compare_arxiv as base
    from benchmark import compare_implementations as helper

    base.ROOT = ROOT
    base.METHODS = ("gecc",)
    helper.REFERENCES["gecc"] = REFERENCE
    original_args = base.configured_args
    author_classes = {}

    def author_class():
        if "GECC" not in author_classes:
            package = importlib.import_module("graphslim.condensation.gecc")
            parent = importlib.import_module("graphslim.condensation.gcond_base")
            namespace = dict(vars(parent))
            reference = folder / "references/GECC/graphslim/condensation"
            helper.definitions(reference / "gcond_base.py", namespace)
            gecc_namespace = dict(vars(package), GCondBase=namespace["GCondBase"])
            helper.definitions(reference / "gecc.py", gecc_namespace)
            author_classes["GECC"] = gecc_namespace["GECC"]
        return author_classes["GECC"]

    def configured_args(folder, job):
        args = original_args(folder, job)
        args.evolve = False
        args.si = "1"
        if job["backend"] == "author":
            author_class()
        return args

    def reduce_graph(folder, job, graph, args):
        import torch
        from graphslim.reduction import create_reducer

        graph.train_mask = graph.idx_train
        if job["backend"] == "author":
            reducer = author_class()("trans", graph, args)
        else:
            module = importlib.import_module("graphslim.condensation.gecc")

            def uniform_centers(new_data, old_centers, needed, device):
                assert old_centers is None, "Only the non-evolving protocol is measured"
                return new_data[torch.randperm(len(new_data))[:needed]].clone()

            module.incremental_kmeanspp_init = uniform_centers
            reducer = create_reducer("gecc", setting="trans", data=graph, args=args)
        reduced = reducer.reduce(graph, verbose=False, save=False)
        return reduced.feat_syn, reduced.adj_syn, reduced.labels_syn

    base.configured_args = configured_args
    base.reduce_graph = reduce_graph
    return base


def plan(options, base):
    base.plan(options)
    folder = options.run_dir.resolve()
    shutil.copy2(__file__, folder / "runner.py")
    shutil.copy2(ROOT / "benchmark/compare_arxiv.py", folder / "source/benchmark/compare_arxiv.py")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["protocol"].pop("cgc")
    manifest["protocol"].pop("clustgdd")
    manifest["protocol"].update(
        precision="float32; TF32 disabled",
        gecc="full transductive graph; depth2; hop weights [0.6,0.5,0.0]; hard clustering; one restart, max300, epsilon1e-8; balance_alpha1",
        initialization="Author uniform sampled centers on both sides; GraphSlim initializer replaced only in this measured adapter. Native package defaults are not benchmarked.",
        adapters=[
            "shared CPU inputs, official splits, node budgets and output contract instead of repository loaders",
            "author GECC and GCondBase definitions loaded verbatim against shared runtime dependencies",
            "training-index alias supplied to both reducers",
            "GraphSlim initialization harmonized to author uniform sampling",
            "author dense diagnostic assignments retained; GraphSlim diagnostic assignments remain sparse and use graph node IDs",
        ],
    )
    previous = json.loads(
        (ROOT / "benchmark/results/arxiv_clustering_comparison_20261101.json").read_text()
    )
    assert manifest["input_sha256"] == previous["source_manifest"]["input_sha256"]
    assert manifest["gpu"]["uuid"] == previous["source_manifest"]["gpu"]["uuid"]
    manifest["paired_existing_cohort"] = dict(
        path="benchmark/results/arxiv_clustering_comparison_20261101.json",
        sha256=base.sha(ROOT / "benchmark/results/arxiv_clustering_comparison_20261101.json"),
        same_input_tensors=True,
        same_gpu_uuid=True,
    )
    manifest["files"] = {
        str(p.relative_to(folder)): base.sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts
    }
    base.write_json(folder / "manifest.json", manifest)


def freeze(options, base):
    import numpy as np
    import psutil
    import sklearn
    import torch

    folder = options.run_dir.resolve()
    summary = base.summarize(folder)
    assert len(summary["records"]) == 18 and all(
        r["status"] == "complete" for r in summary["records"]
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert base.sha(folder / name) == expected, name
    parity = []
    for rate in manifest["rates"]:
        for seed in manifest["seeds"]:
            prefix = f"gecc__r{rate}__s{seed}__"
            a, b = [
                torch.load(folder / "jobs" / f"{prefix}{backend}" / "reduced.pt", weights_only=True)
                for backend in ("author", "graphslim")
            ]
            differences = [float((x - y).abs().max()) for x, y in zip(a, b)]
            assert all(torch.equal(x, y) for x, y in zip(a, b)), (rate, seed, differences)
            parity.append(dict(rate=rate, seed=seed, exact=True, max_abs=differences))
    target = options.snapshot.resolve()
    archive = target.with_name("gecc_arxiv_source_20261102.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for p in sorted((folder / "source").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                bundle.write(p, str(p.relative_to(folder)))
        bundle.write(folder / "runner.py", "runner.py")
        bundle.write(folder / "manifest.json", "manifest.json")
    base.write_json(
        target,
        dict(
            summary,
            exported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            execution_environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                cuda_runtime=torch.version.cuda,
                numpy=np.__version__,
                sklearn=sklearn.__version__,
                psutil=psutil.__version__,
            ),
            source_run=str(folder.relative_to(ROOT)),
            source_manifest=manifest,
            source_archive=dict(path=str(archive.relative_to(ROOT)), sha256=base.sha(archive)),
            result_sha256={
                r["id"]: base.sha(folder / "jobs" / r["id"] / "result.json")
                for r in summary["records"]
            },
            paired_output_parity=parity,
            parity_scope="Reduced features, adjacency and labels; diagnostic assignments differ in storage and indexing.",
        ),
    )
    print(
        f"Frozen {len(summary['records'])} complete GECC reductions; exact paired outputs for all nine rate/seed pairs."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "worker", "summarize", "freeze"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["gecc"], choices=["gecc"])
    parser.add_argument("--rates", nargs="+", default=["0.001", "0.005", "0.01"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpu-index", type=int, default=2)
    parser.add_argument("--gpu-uuid", default="GPU-63b110a5-9b61-48c4-cc62-ec3b3010f952")
    parser.add_argument("--job")
    parser.add_argument("--job-limit", type=int)
    parser.add_argument("--package-source", type=Path)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "benchmark/results/gecc_arxiv_comparison_20261102.json",
    )
    options = parser.parse_args()
    base = configure(options.run_dir.resolve())
    if options.action == "plan":
        plan(options, base)
    elif options.action == "freeze":
        freeze(options, base)
    elif options.action == "summarize":
        base.summarize(options.run_dir.resolve())
    else:
        getattr(base, options.action)(options)


if __name__ == "__main__":
    main()

"""Paired numerical, latency and memory checks for clustering operators.

Microbenchmarks are not end-to-end Products speed claims. Compare complete
reducers on small fixed graphs against an explicitly sealed source snapshot.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import BisectingKMeans
from torch_sparse import SparseTensor, matmul

from benchmark import rerun_nr as nr
from graphslim.reduction.cluster_ops import (
    fitted_labels,
    nearest_squared,
    softmax_cluster_centers,
    sparse_membership,
    weighted_propagation,
)

ROOT = Path(__file__).resolve().parents[1]


def close(left, right, *, exact=False, atol=1e-5, rtol=1e-5):
    if isinstance(left, (tuple, list)):
        for a, b in zip(left, right, strict=True):
            close(a, b, exact=exact, atol=atol, rtol=rtol)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(
            left, right, atol=0 if exact else atol, rtol=0 if exact else rtol
        )
    else:
        np.testing.assert_allclose(
            left, right, atol=0 if exact else atol, rtol=0 if exact else rtol
        )


def measure(function, cuda):
    timings, peaks = [], []
    output = function()
    if cuda:
        torch.cuda.synchronize()
    del output
    for _ in range(3):
        if cuda:
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = function()
        if cuda:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated() - baseline if cuda else None)
        del output
    return dict(
        seconds=timings,
        median_seconds=statistics.median(timings),
        peak_additional_gpu_bytes=max(peaks) if cuda else None,
    )


def check_operators(device):
    cuda = device.startswith("cuda")
    results = []

    def paired(name, original, optimized, **kwargs):
        close(original(), optimized(), **kwargs)
        before, after = measure(original, cuda), measure(optimized, cuda)
        results.append(
            dict(
                operator=name,
                passed=True,
                reference=before,
                optimized=after,
                speedup=before["median_seconds"] / after["median_seconds"],
            )
        )

    torch.manual_seed(7)
    points = torch.randn(4096, 64, device=device)
    centers = torch.randn(128, 64, device=device)
    paired(
        "GECC bounded squared distances",
        lambda: (points[:, None] - centers[None]).square().sum(2).min(1),
        lambda: nearest_squared(points, centers),
        exact=True,
    )
    values = torch.randn(32768, 64, device=device)
    scores = torch.randn(32768, device=device)
    labels = torch.arange(32768, device=device) % 128
    paired(
        "CGC segmented weighted centers",
        lambda: torch.stack(
            [
                (values[labels == c] * scores[labels == c].softmax(0)[:, None]).sum(0)
                for c in range(128)
            ]
        ),
        lambda: softmax_cluster_centers(values, scores, labels, 128),
    )
    n = 2048
    rows = torch.arange(n, device=device).repeat_interleave(16)
    columns = torch.randint(n, (len(rows),), device=device)
    adjacency = SparseTensor(
        row=rows,
        col=columns,
        value=torch.full((len(rows),), 1 / 16, device=device),
        sparse_sizes=(n, n),
    ).coalesce()
    features = torch.randn(n, 64, device=device)
    paired(
        "ClusteringAgg two sparse propagations",
        lambda: matmul(matmul(adjacency, adjacency), features),
        lambda: weighted_propagation(adjacency, features, (0, 0, 1)),
    )

    def old_hops():
        hops = [features]
        for _ in range(4):
            hops.append(matmul(adjacency, hops[-1]).float())
        result = torch.zeros_like(features)
        for weight, hop in zip([0.9, 0.1, 0.2, 0.5, 0.5], hops):
            result.add_(weight * hop)
        return result

    paired(
        "GECC streamed weighted hops",
        old_hops,
        lambda: weighted_propagation(adjacency, features, [0.9, 0.1, 0.2, 0.5, 0.5]),
        exact=True,
    )

    # The following operators deliberately retain native CPU sklearn/SciPy.
    cuda = False
    rng = np.random.default_rng(4)
    data = rng.normal(size=(10000, 32)).astype("float32")

    def fit_predict_twice():
        model = BisectingKMeans(n_clusters=32, random_state=0)
        model.fit(data)
        return model.predict(data)

    paired(
        "Clustering reuse fitted labels",
        fit_predict_twice,
        lambda: fitted_labels(BisectingKMeans(n_clusters=32, random_state=0), data),
        exact=True,
    )
    groups = np.arange(20000) % 128
    weights = rng.integers(1, 100, size=len(groups))
    data = rng.normal(size=(len(groups), 32))

    def dense_membership():
        matrix = np.zeros((128, len(groups)))
        matrix[groups, np.arange(len(groups))] = weights
        matrix /= matrix.sum(1, keepdims=True)
        return matrix @ data

    paired(
        "VNG sparse weighted membership",
        dense_membership,
        lambda: sparse_membership(groups, 128, weights, normalize=True) @ data,
        atol=1e-12,
        rtol=1e-12,
    )
    return results


def check_reducers(reference, output):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_lightweight_methods import tiny_graph, quick_args
    from graphslim.reduction import create_reducer
    from graphslim.coarsening.clusteringagg import ClusterAgg
    from graphslim.coarsening.vng import VNG
    from graphslim.utils import seed_everything

    specification = {
        "gecc": ("graphslim/condensation/gecc.py", "GECC"),
        "cgc": ("graphslim/condensation/cgc.py", "CGC"),
        "clustgdd": ("graphslim/condensation/clustgdd.py", "ClustGDD"),
        "clustering": ("graphslim/coarsening/clustering.py", "Cluster"),
        "clusteringagg": ("graphslim/coarsening/clusteringagg.py", "ClusterAgg"),
        "vng": ("graphslim/coarsening/vng.py", "VNG"),
    }
    manifest = nr.read_json(reference / "manifest.json")
    assert (
        hashlib.sha256((reference / "source.zip").read_bytes()).hexdigest()
        == manifest["archive_sha256"]
    )
    checks = []
    with zipfile.ZipFile(reference / "source.zip") as archive:
        for method, (relative, class_name) in specification.items():
            path = reference / "frozen_workspace" / relative
            assert path.read_bytes() == archive.read(relative)
            spec = importlib.util.spec_from_file_location(f"reference_{method}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            args = quick_args(
                output / "scratch", "clustering" if method == "clusteringagg" else method, "trans"
            )
            args.fuzziness, args.rep_fuzz, args.balance_alpha = 1.0, 1, 1
            args.agg_alpha, args.agg_beta, args.agg_gamma, args.depth = 0.1, 0.2, 0.9, 2
            graph = tiny_graph()
            if method == "vng":
                generator = torch.Generator().manual_seed(8)
                embeds = [torch.randn(16, 12, generator=generator)]
                labels = np.eye(2)[graph.labels_train.numpy()]
                old = module.VNG("trans", graph, args).vng(
                    graph.feat_train, embeds, graph.adj_train, labels
                )
                new = VNG("trans", graph, args).vng(
                    graph.feat_train, embeds, graph.adj_train, labels
                )
            else:
                seed_everything(9)
                source = deepcopy(graph)
                before = getattr(module, class_name)("trans", source, args).reduce(
                    source, verbose=False, save=False
                )
                seed_everything(9)
                source = deepcopy(graph)
                reducer = (
                    ClusterAgg("trans", source, args)
                    if method == "clusteringagg"
                    else create_reducer(method, "trans", source, args)
                )
                after = reducer.reduce(source, verbose=False, save=False)
                old = (before.adj_syn, before.feat_syn, before.labels_syn)
                new = (after.adj_syn, after.feat_syn, after.labels_syn)
            close(old, new, atol=1e-4, rtol=1e-4)
            checks.append(
                dict(
                    method=method,
                    passed=True,
                    atol=1e-4,
                    rtol=1e-4,
                    max_abs=[float((a - b).abs().max()) for a, b in zip(old, new)],
                )
            )
    return dict(
        reference_source=manifest["source_sha256"],
        checks=checks,
        scope="Fixed small-graph reduction checks; Products GECC propagation repair is separately tested, not equivalent to its legacy sampled branch",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("runs/products_streaming_20261028"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = dict(
        utc=nr.utcnow(),
        device=args.device,
        source_sha256=nr.source_digest(),
        operators=check_operators(args.device),
        reducers=check_reducers(args.reference, args.output),
        scope="Three timed repeats after warmup; isolated operators, not Products end-to-end speedups. Sparse/segmented results use tolerance; distance ties and GECC hops are exact.",
    )
    nr.write_json(args.output / "validation.json", report)
    print(
        json.dumps(
            [{k: r[k] for k in ("operator", "passed", "speedup")} for r in report["operators"]]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

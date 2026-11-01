"""Matched, isolated author/GraphSlim reduction measurements.

The comparison uses frozen source and common in-memory inputs and GCN evaluation.
It is not a comparison of the original repositories' complete command-line runs.
"""

import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import fcntl
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file() and (parent / "graphslim").is_dir()
)
REFERENCES = {
    "cgc": ("CGC", "08ff04dc0d23110ff53445438f30bcff887e7c31", "https://github.com/XYGaoG/CGC"),
    "clustgdd": (
        "ClustGDD",
        "60825c536f794e7a3bb0ff22ae4efc07ea2d3e3a",
        "https://github.com/HKBU-LAGAS/ClustGDD",
    ),
    "gcgp": (
        "GCGP",
        "d81ed427cd52d5a0d2a74bdf4328d31db1dcc1eb",
        "https://github.com/WANGLin0126/GCGP",
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def definitions(path, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plan(options):
    folder = options.run_dir.resolve()
    if (folder / "manifest.json").exists():
        raise ValueError("Use a new run directory; existing scientific manifests are immutable")
    snapshot = folder / "source"
    snapshot.mkdir(parents=True)
    shutil.copytree(
        ROOT / "graphslim",
        snapshot / "graphslim",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(__file__, folder / "runner.py")
    references = {}
    for method in options.methods:
        name, commit, url = REFERENCES[method]
        source = ROOT / "runs" / "reference" / name
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise ValueError(f"Unpinned author reference: {name} {actual}")
        subprocess.run(
            ["git", "-C", str(source), "diff", "--exit-code", "HEAD"],
            check=True,
            capture_output=True,
        )
        shutil.copytree(
            source,
            folder / "references" / name,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "datasets", "data", "*.pyc"),
        )
        references[method] = dict(commit=commit, url=url)
    files = {
        str(p.relative_to(folder)): digest(p)
        for base in (snapshot, folder / "references")
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }
    files["runner.py"] = digest(folder / "runner.py")
    jobs = []
    for method in options.methods:
        for dataset in options.datasets:
            for seed in options.seeds:
                # Alternate order across seeds to reduce systematic thermal/order effects.
                for backend in ("author", "graphslim") if seed % 2 else ("graphslim", "author"):
                    jobs.append(
                        dict(
                            id=f"{method}__{dataset}__s{seed}__{backend}",
                            method=method,
                            dataset=dataset,
                            seed=seed,
                            backend=backend,
                        )
                    )
    manifest = dict(
        schema_version=1,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        references=references,
        files=files,
        methods=options.methods,
        datasets=options.datasets,
        seeds=options.seeds,
        eval_seeds=list(range(options.evaluations)),
        jobs=jobs,
        protocol=dict(
            rate=0.5,
            data="Planetoid public splits; identical saved tensors per method/dataset",
            feature_processing="row-normalized CGC/ClustGDD; raw BOW GCGP",
            reduction_precision="float32; CGC reconstruction alone float64 on both implementations, ridge=0",
            cgc="spectral clustering; author FAISS vs package sklearn; both 10 restarts and at most 300 iterations; equal alpha/tau/augmentation/threshold",
            clustgdd="600 pretraining + 100 refinement updates, hidden 256, 2 layers; equal propagation/dropout/sampling/loss settings",
            gcgp="200 full-batch Adam updates, released SGNK/GP arithmetic, no learned adjacency; same initialization RNG device",
            quality="shared PyG 2-layer GCN, hidden256, 300 epochs, lr0.01, wd0.0005, dropout0; validation every10, test only at chosen checkpoint",
            timing="full in-memory reduction from common CPU tensors through normalized output contract; includes preprocessing and device transfers; excludes imports, input loading, serialization, downstream GCN and diagnostic GP test inference",
            memory="fresh process per job; sampled process RSS every5ms during reduction; CUDA peak allocated/reserved reset after context warmup; resident input/import baseline and increments retained",
            adapters=[
                "common data/split injection replaces each repository loader",
                "CGC reconstruction precision harmonized to float64, both return float32",
                "CGC FAISS and ClustGDD sklearn restart/iteration budgets explicitly matched to package clustering (10 x 300)",
                "ClustGDD graph_compress receives integer cluster IDs (released caller supplies floats)",
                "ClustGDD output uses unnormalized compressed adjacency for common GCN normalization",
                "GCGP trains with the author train() definition; per-epoch test probes omitted from both sides",
                "GCGP optimized target logits are softmaxed for shared downstream probability-regression loss",
            ],
            scope="new controlled reduction comparison, not paper-score reproduction or an unmodified author CLI benchmark; no all-method claim",
        ),
    )
    write_json(folder / "manifest.json", manifest)
    print(f"Planned {len(jobs)} isolated jobs in {folder}")


def load_inputs(folder, method, dataset):
    import numpy as np
    import torch
    from graphslim.config import get_args
    from graphslim.dataset import get_dataset
    from graphslim.reduction.lightweight import as_csr

    target = folder / "inputs" / f"{method}_{dataset}.pt"
    if target.exists():
        return torch.load(target, weights_only=True)
    with redirect_stdout(io.StringIO()):
        args = get_args(
            ["-D", dataset, "-M", method, "-G", "-1", "--save_path", str(folder / "scratch")]
        )
        args.pre_norm = False
        data = get_dataset(dataset, args, str(ROOT / "data"))
    x = torch.as_tensor(data.feat_full).float().cpu()
    if method != "gcgp":
        x = x / x.sum(1, keepdim=True).clamp_min(1)
    adjacency = as_csr(data.adj_full).tocoo()
    packed = dict(
        x=x,
        y=torch.as_tensor(data.labels_full).long().cpu(),
        edge_index=torch.from_numpy(np.stack((adjacency.row, adjacency.col))).long(),
        edge_weight=torch.from_numpy(adjacency.data).float(),
        idx_train=torch.as_tensor(data.idx_train).long(),
        idx_val=torch.as_tensor(data.idx_val).long(),
        idx_test=torch.as_tensor(data.idx_test).long(),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, target)
    return packed


def make_graph(packed):
    import scipy.sparse as sp
    import torch

    x, y = packed["x"], packed["y"]
    train, val, test = (packed[f"idx_{part}"] for part in ("train", "val", "test"))
    adj = sp.csr_matrix(
        (packed["edge_weight"].numpy(), packed["edge_index"].numpy()), shape=(len(x), len(x))
    )
    return SimpleNamespace(
        feat_full=x,
        adj_full=adj,
        labels_full=y,
        feat_train=x[train],
        adj_train=adj[train.numpy()][:, train.numpy()],
        labels_train=y[train],
        idx_train=train.numpy(),
        idx_val=val.numpy(),
        idx_test=test.numpy(),
        labels_val=y[val],
        labels_test=y[test],
        feat_val=x[val],
        nclass=int(y.max()) + 1,
        feat_syn=None,
        adj_syn=None,
        labels_syn=None,
    )


def configured_args(folder, job, device):
    from graphslim.config import get_args

    with redirect_stdout(io.StringIO()):
        args = get_args(
            [
                "-D",
                job["dataset"],
                "-M",
                job["method"],
                "-G",
                "-1",
                "-R",
                "0.5",
                "--epochs",
                "100" if job["method"] == "clustgdd" else "200",
                "--save_path",
                str(folder / "scratch"),
            ]
        )
    args.device = device
    args.seed = job["seed"]
    args.clustgdd_pre_epochs = 600
    args.clustgdd_hidden = 256
    args.clustgdd_post_steps = 2
    args.gcgp_arithmetic = "released"
    args.gcgp_batch_size = 100000
    return args


def author_cgc(folder, graph, args):
    import faiss
    import numpy as np
    import scipy.sparse as sp
    import torch
    import torch.nn.functional as F
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from torch_geometric.data import Data

    native_kmeans = faiss.Kmeans
    matched_faiss = SimpleNamespace(
        Kmeans=lambda *a, **kw: native_kmeans(*a, **kw, niter=300, nredo=10, seed=args.seed)
    )
    namespace = dict(
        torch=torch,
        np=np,
        sp=sp,
        F=F,
        faiss=matched_faiss,
        TruncatedSVD=TruncatedSVD,
        classification_report=classification_report,
        PolynomialFeatures=PolynomialFeatures,
        StandardScaler=StandardScaler,
    )
    namespace = definitions(folder / "references/CGC/scr/utils.py", namespace)
    mask = torch.zeros(len(graph.feat_full), dtype=torch.bool)
    mask[graph.idx_train] = True
    coo = graph.adj_full.tocoo()
    data = Data(
        x=graph.feat_full.clone(),
        y=graph.labels_full.clone(),
        train_mask=mask,
        edge_index=torch.tensor(np.stack((coo.row, coo.col))),
    )
    config = SimpleNamespace(
        dataset_name=args.dataset,
        ratio=0.026 if args.dataset == "cora" else 0.018,
        num_class=graph.nclass,
        kernel="gcn",
        device="cpu",
        aug_ratio=args.cgc_aug_ratio,
        tau=args.cgc_tau,
        adj_T=args.cgc_adj_threshold,
        alpha=args.cgc_alpha,
    )
    config, labels = namespace["generate_labels_syn"](config, data)
    hops = namespace["conv_graph_multi"](config, data)
    linear = namespace["linear_model"](config, hops, data, None)
    augmented, y, confidence = namespace["data_assessment"](config, data, linear, hops)
    mapping = namespace["mask_generation_conf"](augmented, y, config, "spectral", confidence)
    centers = torch.sparse.mm(mapping, augmented)
    adjacency = namespace["get_adj"](centers, config.adj_T)
    features = namespace["get_feature"](adjacency.double(), centers.double(), config.alpha).float()
    return features, adjacency, labels


def author_gcgp(folder, graph, args):
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F

    common = dict(torch=torch, nn=nn, F=F, np=np, random=random)
    base = folder / "references/GCGP/gcgp"
    kernel = definitions(base / "sgnk.py", dict(common))["SimplifyingGraphNeuralKernel"](L=2).to(
        args.device
    )
    gp = definitions(base / "gpr.py", dict(common))["GaussianProcessRegression"](
        kernel.nodes_gram, args.gcgp_ridge, args.gcgp_k
    ).to(args.device)
    load = definitions(base / "LoadData.py", dict(common))
    # Author preprocessing materializes a dense graph; package uses sparse propagation.
    adjacency = torch.from_numpy(graph.adj_full.toarray()).float().to(args.device)
    adjacency = adjacency + torch.eye(len(adjacency), device=args.device)
    filtered = load["GCF"](adjacency, graph.feat_full.to(args.device), args.gcgp_k)[graph.idx_train]
    target = F.one_hot(graph.labels_train.to(args.device), graph.nclass).float()
    size = round(len(graph.idx_train) * args.reduction_rate)
    xs = torch.rand(size, filtered.shape[1], device=args.device, requires_grad=True)
    ys = torch.rand(size, graph.nclass, device=args.device, requires_grad=True)
    alpha = 2 * torch.rand(size, size, device=args.device)
    identity = torch.eye(size, device=args.device)
    optimizer = torch.optim.Adam([xs, ys], lr=args.gcgp_lr)
    space = definitions(
        base / "main.py",
        dict(common, GPR=gp, optimizer_X=optimizer, optimizer_A=None, x_s=xs, y_s=ys),
    )
    loss_function = nn.MSELoss()
    for epoch in range(args.epochs):
        space["train"](filtered, xs, target, ys, None, identity, alpha, loss_function, epoch, False)
    return xs.detach(), identity, ys.detach().softmax(1)


def author_clustgdd(folder, graph, args):
    import numpy as np
    import torch

    sys.path.insert(0, str(folder / "references/ClustGDD"))
    module = importlib.import_module("clustgdd_agent_transduct")
    from functools import partial
    from sklearn.cluster import KMeans

    module.KMeans = partial(KMeans, n_init=10, max_iter=300, tol=1e-4, random_state=args.seed)
    data = deepcopy(graph)
    for name in (
        "feat_full",
        "feat_train",
        "labels_full",
        "labels_train",
        "labels_val",
        "labels_test",
    ):
        setattr(data, name, getattr(data, name).numpy())
    config = SimpleNamespace(
        dataset=args.dataset,
        reduction_rate=args.reduction_rate,
        seed=args.seed,
        hidden=256,
        prop_num=args.clustgdd_prop_steps,
        alpha=args.clustgdd_alpha,
        predropout=args.clustgdd_dropout,
        prewd=0.0,
        prelr=args.clustgdd_pre_lr,
        preep=600,
        prenlayers=2,
        cluster_minibatch=1000,
        sp_ratio=args.clustgdd_sampling_ratio,
        sp_type="attaw",
        postep=args.epochs,
        postprop_num=2,
        postlr_feat=args.clustgdd_feature_lr,
        postwd_feat=args.clustgdd_post_wd,
        postlr_model=args.clustgdd_post_lr,
        postwd_model=args.clustgdd_post_wd,
        frcoe=args.clustgdd_refinement_scale,
        predcoe=args.clustgdd_prediction_weight,
        csttemp=args.clustgdd_temperature,
        w1=args.clustgdd_consistency_weight,
        w2=1.0,
    )
    agent = module.ClustGDD(data, config, device=args.device)
    feat, labels, clusters, normalized, _, _, full_labels, target, train, val, embeddings = (
        agent.pretrained_clustering(data)
    )
    views = agent.graph_sparse(
        normalized, ratio=config.sp_ratio, ebd=embeddings, sp_type=config.sp_type
    )
    compressed, adj = agent.graph_compress(clusters.long(), normalized, views)
    feat = agent.graph_refusion(target, train, val, full_labels, feat, compressed, labels)
    return feat.detach(), adj.to_dense().detach(), labels.detach()


def reduce_graph(folder, job, graph, args):
    if job["backend"] == "author":
        return globals()[f"author_{job['method']}"](folder, graph, args)
    from graphslim.reduction import create_reducer

    if job["method"] == "cgc":
        module = importlib.import_module("graphslim.condensation.cgc")
        original = module.reconstruct_features
        module.reconstruct_features = lambda a, h, alpha: original(
            a.double(), h.double(), alpha, ridge=0
        ).float()
    reducer = create_reducer(job["method"], setting="trans", data=graph, args=args)
    if job["method"] == "gcgp":
        reducer.gp_test_accuracy = lambda *a, **kw: None
    reduced = reducer.reduce(graph, verbose=False, save=False)
    return reduced.feat_syn, reduced.adj_syn, reduced.labels_syn


def quality(packed, output, device, seeds):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    features, adjacency, labels = (t.detach().to(device) for t in output)
    adjacency = adjacency.to_dense() if adjacency.layout != torch.strided else adjacency
    indices = adjacency.nonzero().T
    values = adjacency[indices[0], indices[1]]
    x, y = packed["x"].to(device), packed["y"].to(device)
    edges, weights = packed["edge_index"].to(device), packed["edge_weight"].to(device)
    val, test = packed["idx_val"].to(device), packed["idx_test"].to(device)
    results = []

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = GCNConv(x.shape[1], 256)
            self.second = GCNConv(256, int(y.max()) + 1)

        def forward(self, features, edges, weights):
            return self.second(F.relu(self.first(features, edges, weights)), edges, weights)

    for seed in seeds:
        seed_all(seed)
        model = GCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        best = -1
        checkpoint = None
        selected_epoch = None
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            logits = model(features, indices, values)
            loss = (
                F.mse_loss(logits.softmax(1), labels)
                if labels.ndim == 2
                else F.cross_entropy(logits, labels)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite shared GCN loss")
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    prediction = model(x, edges, weights).argmax(1)
                    score = float((prediction[val] == y[val]).float().mean())
                if score > best:
                    best = score
                    checkpoint = deepcopy(model.state_dict())
                    selected_epoch = epoch + 1
        model.load_state_dict(checkpoint)
        model.eval()
        with torch.no_grad():
            prediction = model(x, edges, weights).argmax(1)
            score = float((prediction[test] == y[test]).float().mean())
        results.append(
            dict(
                seed=seed,
                validation_accuracy=best,
                test_accuracy=score,
                selected_epoch=selected_epoch,
            )
        )
    return results


def worker(options):
    folder = options.run_dir.resolve()
    sys.path.insert(0, str(folder / "source"))
    sys.path.insert(0, str(ROOT / "runs/implementation_comparison_deps_20261031"))
    import numpy as np
    import psutil
    import torch

    manifest = json.loads((folder / "manifest.json").read_text())
    job = next(j for j in manifest["jobs"] if j["id"] == options.job)
    result = dict(job, status="running")
    destination = folder / "jobs" / job["id"]
    destination.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = "cpu" if job["method"] == "cgc" else "cuda:0"
        packed = torch.load(
            folder / "inputs" / f"{job['method']}_{job['dataset']}.pt", weights_only=True
        )
        graph = make_graph(packed)
        args = configured_args(folder, job, device)
        # Equal import baseline for the two implementations of each method.
        importlib.import_module(f"graphslim.condensation.{job['method']}")
        if job["method"] == "cgc":
            import faiss

            faiss.omp_set_num_threads(4)
        elif job["method"] == "clustgdd":
            sys.path.insert(0, str(folder / "references/ClustGDD"))
            importlib.import_module("clustgdd_agent_transduct")
        if device.startswith("cuda"):
            torch.cuda.set_device(0)
            warm = torch.rand(256, 256, device=device)
            _ = warm @ warm
            del warm, _
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        seed_all(job["seed"])
        process = psutil.Process()
        baseline = process.memory_info().rss
        samples = [baseline]
        stopped = threading.Event()

        def sample():
            while not stopped.wait(0.005):
                samples.append(process.memory_info().rss)

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        start = time.perf_counter()
        try:
            output = reduce_graph(folder, job, graph, args)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        finally:
            samples.append(process.memory_info().rss)
            stopped.set()
            monitor.join()
        memory = dict(
            rss_baseline_bytes=baseline,
            peak_sampled_rss_bytes=max(samples),
            additional_sampled_rss_bytes=max(samples) - baseline,
            rss_sample_interval_seconds=0.005,
        )
        if device.startswith("cuda"):
            memory.update(
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        nodes = len(output[0])
        assert nodes == int(len(graph.idx_train) * 0.5), (nodes, len(graph.idx_train))
        assert all(torch.isfinite(t).all() for t in output)
        result.update(
            reduction_seconds=seconds,
            memory=memory,
            synthetic_nodes=nodes,
            device=device,
            input_sha256={key: tensor_hash(value) for key, value in packed.items()},
            output_sha256=[tensor_hash(t) for t in output],
            settings={
                k: v
                for k, v in vars(args).items()
                if k.startswith(("cgc_", "clustgdd_", "gcgp_"))
                or k in ("epochs", "seed", "reduction_rate")
            },
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                numpy=np.__version__,
                cpu=platform.processor(),
                threads=4,
                gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                tf32=False,
            ),
        )
        torch.save(tuple(t.detach().cpu() for t in output), destination / "reduced.pt")
        result["status"] = "reduced"
        write_json(destination / "result.json", result)
        result["evaluations"] = quality(packed, output, device, manifest["eval_seeds"])
        result["status"] = "complete"
        write_json(destination / "result.json", result)
        print(
            f"COMPLETE {job['id']} time={seconds:.3f}s accuracy={100 * statistics.mean(e['test_accuracy'] for e in result['evaluations']):.2f}",
            flush=True,
        )
    except Exception as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        write_json(destination / "result.json", result)
        print(result["traceback"], flush=True)
        raise


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = []
    for method in manifest["methods"]:
        for dataset in manifest["datasets"]:
            paired = {}
            for backend in ("author", "graphslim"):
                records = []
                for seed in manifest["seeds"]:
                    path = (
                        folder / "jobs" / f"{method}__{dataset}__s{seed}__{backend}" / "result.json"
                    )
                    records.append(
                        json.loads(path.read_text()) if path.exists() else dict(status="pending")
                    )
                complete = all(
                    r["status"] == "complete"
                    and len(r.get("evaluations", [])) == len(manifest["eval_seeds"])
                    for r in records
                )
                row = dict(
                    method=method,
                    dataset=dataset,
                    backend=backend,
                    status="complete" if complete else "incomplete",
                    completed_seeds=sum(r["status"] == "complete" for r in records),
                    expected_seeds=len(records),
                )
                if complete:
                    scores = [100 * e["test_accuracy"] for r in records for e in r["evaluations"]]
                    row.update(
                        reduction_seconds_median=statistics.median(
                            r["reduction_seconds"] for r in records
                        ),
                        peak_rss_mib_median=statistics.median(
                            r["memory"]["peak_sampled_rss_bytes"] / 2**20 for r in records
                        ),
                        additional_rss_mib_median=statistics.median(
                            r["memory"]["additional_sampled_rss_bytes"] / 2**20 for r in records
                        ),
                        peak_cuda_mib_median=statistics.median(
                            r["memory"].get("peak_cuda_allocated_bytes", 0) / 2**20 for r in records
                        ),
                        accuracy_mean=statistics.mean(scores),
                        accuracy_std=statistics.pstdev(scores),
                        evaluations=len(scores),
                        device=records[0]["device"],
                        synthetic_nodes=records[0]["synthetic_nodes"],
                    )
                    paired[backend] = records
                rows.append(row)
            if len(paired) == 2:
                for a, b in zip(paired["author"], paired["graphslim"]):
                    if (
                        a["input_sha256"] != b["input_sha256"]
                        or a["synthetic_nodes"] != b["synthetic_nodes"]
                    ):
                        raise ValueError("Mismatched paired inputs/budget")
                for row in rows[-2:]:
                    row["author_over_graphslim_time"] = statistics.median(
                        r["reduction_seconds"] for r in paired["author"]
                    ) / statistics.median(r["reduction_seconds"] for r in paired["graphslim"])
    write_json(folder / "summary.json", dict(protocol=manifest["protocol"], rows=rows))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(options):
    folder = options.run_dir.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(folder / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    sys.path.insert(0, str(folder / "source"))
    for method in manifest["methods"]:
        for dataset in manifest["datasets"]:
            load_inputs(folder, method, dataset)
    locks = []
    try:
        for path in (
            f"/tmp/ekpt-physical-gpu-{options.gpu_index}.lock",
            f"/tmp/graphslim-{options.gpu_uuid}.lock",
        ):
            handle = open(path, "a+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        for job in manifest["jobs"]:
            path = folder / "jobs" / job["id"] / "result.json"
            if path.exists():
                continue
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
            )
            if options.gpu_uuid in apps:
                raise RuntimeError("Selected GPU is occupied; existing jobs left untouched")
            target = folder / "jobs" / job["id"]
            target.mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=options.gpu_uuid,
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                MPLCONFIGDIR=str(folder / "mpl"),
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
            )
            command = [
                sys.executable,
                str(folder / "runner.py"),
                "worker",
                "--run-dir",
                str(folder),
                "--job",
                job["id"],
            ]
            print("START", job["id"], flush=True)
            with (target / "run.log").open("w") as log:
                process = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800
                )
            if process.returncode:
                print("FAILED", job["id"], "see", target / "run.log", flush=True)
            else:
                record = json.loads(path.read_text())
                print("DONE", job["id"], f"{record['reduction_seconds']:.3f}s", flush=True)
            summarize(folder)
    finally:
        for handle in locks:
            handle.close()
    summarize(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "worker", "summarize"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(REFERENCES), choices=list(REFERENCES))
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--gpu-index", type=int, default=2)
    parser.add_argument("--gpu-uuid", default="GPU-63b110a5-9b61-48c4-cc62-ec3b3010f952")
    parser.add_argument("--job")
    options = parser.parse_args()
    if options.action == "summarize":
        summarize(options.run_dir.resolve())
    else:
        globals()[options.action](options)


if __name__ == "__main__":
    main()

"""Measure matched author/GraphSlim clustering reducers on full ogbn-arxiv.

Plan freezes package/author sources and one common standardized graph. Run each
reduction in a fresh process on one UUID-bound GPU, then evaluate its saved graph
outside timing. Historical Cora/Citeseer measurements are never pooled here.
"""

import argparse
import ast
from contextlib import redirect_stdout
import fcntl
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
METHODS = ("cgc", "clustgdd")
RATES = ("0.001", "0.005", "0.01")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs(methods=METHODS, rates=RATES, seeds=(1, 2, 3)):
    return [
        dict(
            id=f"{method}__r{rate}__s{seed}__{backend}",
            method=method,
            dataset="ogbn-arxiv",
            rate=rate,
            seed=seed,
            backend=backend,
        )
        for rate in rates
        for method in methods
        for seed in seeds
        for backend in (("author", "graphslim") if seed % 2 else ("graphslim", "author"))
    ]


def plan(options):
    import numpy as np
    import scipy.sparse as sp
    import torch
    from sklearn.preprocessing import StandardScaler
    from benchmark.compare_implementations import REFERENCES, tensor_hash

    folder = options.run_dir.resolve()
    if folder.exists():
        raise ValueError("Use a new run directory; frozen manifests are immutable")
    folder.mkdir(parents=True)
    source = folder / "source"
    package = (options.package_source or ROOT / "graphslim").resolve()
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Invalid GraphSlim source directory: {package}")
    shutil.copytree(
        package,
        source / "graphslim",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (source / "benchmark").mkdir()
    shutil.copy2(ROOT / "benchmark/__init__.py", source / "benchmark/__init__.py")
    helper = package.parent / "benchmark/compare_implementations.py"
    shutil.copy2(
        helper if helper.is_file() else ROOT / "benchmark/compare_implementations.py",
        source / "benchmark/compare_implementations.py",
    )
    project = package.parent / "pyproject.toml"
    shutil.copy2(
        project if project.is_file() else ROOT / "pyproject.toml", source / "pyproject.toml"
    )
    shutil.copy2(__file__, folder / "runner.py")
    references = {}
    for method in options.methods:
        name, commit, url = REFERENCES[method]
        reference = ROOT / "runs/reference" / name
        assert (
            subprocess.check_output(
                ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
            ).strip()
            == commit
        )
        subprocess.run(
            ["git", "-C", str(reference), "diff", "--exit-code", "HEAD"],
            check=True,
            capture_output=True,
        )
        shutil.copytree(
            reference,
            folder / "references" / name,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "data", "datasets", "*.pyc"),
        )
        references[method] = dict(commit=commit, url=url)
    raw = ROOT / "data/ogbn_arxiv/raw"
    roles = json.loads((raw / "role.json").read_text())
    features = np.load(raw / "feats.npy").astype(np.float32)
    scaler = StandardScaler().fit(features[roles["tr"]])
    features = scaler.transform(features)
    adjacency = sp.load_npz(raw / "adj_full.npz").astype(np.float32)
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()
    coo = adjacency.tocoo()
    labels = json.loads((raw / "class_map.json").read_text())
    labels = np.array([labels[str(i)] for i in range(len(features))], dtype=np.int64)
    packed = dict(
        x=torch.from_numpy(features),
        y=torch.from_numpy(labels),
        edge_index=torch.from_numpy(np.stack([coo.row, coo.col]).astype(np.int64)),
        edge_weight=torch.from_numpy(coo.data),
        idx_train=torch.tensor(roles["tr"]),
        idx_val=torch.tensor(roles["va"]),
        idx_test=torch.tensor(roles["te"]),
    )
    torch.save(packed, folder / "inputs.pt")
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    manifest = dict(
        schema_version=1,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        methods=options.methods,
        rates=options.rates,
        seeds=options.seeds,
        jobs=jobs(options.methods, options.rates, options.seeds),
        references=references,
        files=files,
        input_sha256={key: tensor_hash(value) for key, value in packed.items()},
        data=dict(
            nodes=len(features),
            edges=adjacency.nnz,
            features=features.shape[1],
            classes=int(labels.max()) + 1,
            train=len(roles["tr"]),
            val=len(roles["va"]),
            test=len(roles["te"]),
            raw_sha256={
                p.name: sha(p)
                for p in [
                    raw / "feats.npy",
                    raw / "adj_full.npz",
                    raw / "role.json",
                    raw / "class_map.json",
                ]
            },
        ),
        gpu=dict(index=options.gpu_index, uuid=options.gpu_uuid),
        protocol=dict(
            threads=4,
            device="cuda:0",
            rates_denominator="labeled training nodes",
            preprocessing="full undirected ogbn-arxiv; StandardScaler fitted on official training nodes, shared saved tensors",
            precision="float32; both CGC reconstruction solves float64, ridge=0; TF32 disabled",
            cgc="rate-specific package parameters; spectral clustering; FAISS/sklearn both 10 restarts, 300 iterations",
            clustgdd="rate-specific propagation/alpha/loss settings; 600 MLP pretraining and 100 refinement updates, hidden256, 2 post-propagation steps; both MiniBatchKMeans n_init3, max_iter100, batch_size1000",
            adapters=[
                "shared inputs/splits/output contract instead of repository loaders",
                "CGC confidence masks copied to CPU for its CPU confidence tensors; native CPU clustering and GPU aggregation retained",
                "CGC reconstruction precision harmonized to float64",
                "ClustGDD MiniBatchKMeans restart budget harmonized; integer IDs passed into native compression",
            ],
            timing="wall time with CUDA synchronization around entire in-memory reduction, including preprocessing/transfers; excludes imports, data loading, graph serialization, and downstream evaluation",
            memory="fresh process; process RSS sampled every5ms; PyTorch peak GPU allocated/reserved reset after context warmup, before input transfers; excludes downstream evaluation",
            quality="one shared GCN fit per reduced graph, seed0, 300 epochs, hidden256, validation selection every10; outside timing/memory; descriptive quality shortfall retained",
            timeout_seconds=1800,
            max_rss_gib=96,
        ),
    )
    write_json(folder / "manifest.json", manifest)
    print(
        f"Frozen {len(manifest['jobs'])} jobs on {manifest['data']['nodes']:,} nodes / {adjacency.nnz:,} edges.",
        flush=True,
    )


def configured_args(folder, job):
    from graphslim.config import get_args

    with redirect_stdout(io.StringIO()):
        args = get_args(
            [
                "-D",
                "ogbn-arxiv",
                "-M",
                job["method"],
                "-G",
                "-1",
                "-R",
                job["rate"],
                "--epochs",
                "100",
                "--save_path",
                str(folder / "scratch"),
            ]
        )
    args.device = "cuda:0"
    args.seed = job["seed"]
    args.clustgdd_pre_epochs = 600
    args.clustgdd_hidden = 256
    args.clustgdd_post_steps = 2
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
    namespace = dict(
        torch=torch,
        np=np,
        sp=sp,
        F=F,
        TruncatedSVD=TruncatedSVD,
        classification_report=classification_report,
        PolynomialFeatures=PolynomialFeatures,
        StandardScaler=StandardScaler,
        faiss=SimpleNamespace(
            Kmeans=lambda *a, **kw: native_kmeans(*a, **kw, niter=300, nredo=10, seed=args.seed)
        ),
    )
    path = folder / "references/CGC/scr/utils.py"
    code = path.read_text().replace(
        "conf_pred_aug[y_pred_aug!=y_aug]", "conf_pred_aug[(y_pred_aug!=y_aug).cpu()]"
    )
    code = code.replace("conf_pred[y_pred!=y]", "conf_pred[(y_pred!=y).cpu()]")
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    mask = torch.zeros(len(graph.feat_full), dtype=torch.bool)
    mask[graph.idx_train] = True
    coo = graph.adj_full.tocoo()
    data = Data(
        x=graph.feat_full.clone(),
        y=graph.labels_full.clone(),
        train_mask=mask,
        edge_index=torch.from_numpy(np.stack([coo.row, coo.col])),
    ).to(args.device)
    config = SimpleNamespace(
        dataset_name="ogbn-arxiv",
        ratio=args.reduction_rate,
        num_class=graph.nclass,
        kernel="gcn",
        device=args.device,
        aug_ratio=args.cgc_aug_ratio,
        tau=args.cgc_tau,
        adj_T=args.cgc_adj_threshold,
        alpha=args.cgc_alpha,
    )
    config, labels = namespace["generate_labels_syn"](config, data)
    hops = namespace["conv_graph_multi"](config, data)
    linear = namespace["linear_model"](config, hops, data, None)
    embedded, targets, confidence = namespace["data_assessment"](config, data, linear, hops)
    membership = namespace["mask_generation_conf"](
        embedded.cpu(), targets.cpu(), config, "spectral", confidence.cpu()
    )
    centers = torch.sparse.mm(membership.to(args.device), embedded)
    adjacency = namespace["get_adj"](centers, config.adj_T)
    features = namespace["get_feature"](adjacency.double(), centers.double(), config.alpha).float()
    return features, adjacency, labels.to(args.device)


def reduce_graph(folder, job, graph, args):
    from benchmark.compare_implementations import author_clustgdd

    if job["backend"] == "author":
        if job["method"] == "cgc":
            return author_cgc(folder, graph, args)
        from functools import partial
        from sklearn.cluster import MiniBatchKMeans

        module = importlib.import_module("clustgdd_agent_transduct")
        module.MiniBatchKMeans = partial(MiniBatchKMeans, n_init=3, max_iter=100)
        return author_clustgdd(folder, graph, args)
    from graphslim.reduction import create_reducer

    if job["method"] == "cgc":
        module = importlib.import_module("graphslim.condensation.cgc")
        original = module.reconstruct_features
        module.reconstruct_features = lambda a, h, alpha: original(
            a.double(), h.double(), alpha, ridge=0
        ).float()
    reducer = create_reducer(job["method"], setting="trans", data=graph, args=args)
    reduced = reducer.reduce(graph, verbose=False, save=False)
    return reduced.feat_syn, reduced.adj_syn, reduced.labels_syn


def worker(options):
    folder = options.run_dir.resolve()
    sys.path.insert(0, str(folder / "source"))
    sys.path.insert(0, str(ROOT / "runs/implementation_comparison_deps_20261031"))
    import numpy as np
    import psutil
    import torch
    import faiss
    from benchmark.compare_implementations import make_graph, quality, seed_all, tensor_hash

    manifest = json.loads((folder / "manifest.json").read_text())
    job = next(j for j in manifest["jobs"] if j["id"] == options.job)
    destination = folder / "jobs" / job["id"]
    destination.mkdir(parents=True, exist_ok=True)
    result = dict(job, status="running")
    write_json(destination / "result.json", result)
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        faiss.omp_set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(0)
        assert torch.cuda.is_available()
        packed = torch.load(folder / "inputs.pt", weights_only=True)
        assert {k: tensor_hash(v) for k, v in packed.items()} == manifest["input_sha256"]
        graph = make_graph(packed)
        args = configured_args(folder, job)
        importlib.import_module(f"graphslim.condensation.{job['method']}")
        if job["method"] == "clustgdd":
            sys.path.insert(0, str(folder / "references/ClustGDD"))
            importlib.import_module("clustgdd_agent_transduct")
        warm = torch.rand(256, 256, device="cuda:0")
        warm = warm @ warm
        del warm
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        seed_all(job["seed"])
        baseline_gpu = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        process = psutil.Process()
        baseline = process.memory_info().rss
        samples = [baseline]
        stop = threading.Event()

        def monitor():
            while not stop.wait(0.005):
                samples.append(process.memory_info().rss)

        sampler = threading.Thread(target=monitor, daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            output = reduce_graph(folder, job, graph, args)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        finally:
            samples.append(process.memory_info().rss)
            stop.set()
            sampler.join()
            result["memory"] = dict(
                rss_baseline_bytes=baseline,
                peak_sampled_rss_bytes=max(samples),
                rss_sample_interval_seconds=0.005,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_allocated_baseline_bytes=baseline_gpu,
            )
        assert len(output[0]) == int(len(graph.idx_train) * args.reduction_rate)
        assert all(torch.isfinite(t).all() for t in output)
        result.update(
            status="reduced",
            reduction_seconds=seconds,
            synthetic_nodes=len(output[0]),
            input_sha256=manifest["input_sha256"],
            output_sha256=[tensor_hash(t) for t in output],
            settings={
                k: v
                for k, v in vars(args).items()
                if k.startswith(("cgc_", "clustgdd_")) or k in ("epochs", "seed", "reduction_rate")
            },
            environment=dict(
                torch=torch.__version__,
                numpy=np.__version__,
                gpu=torch.cuda.get_device_name(0),
                gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
                threads=4,
                tf32=False,
            ),
        )
        torch.save(tuple(t.detach().cpu() for t in output), destination / "reduced.pt")
        write_json(destination / "result.json", result)
        print(
            f"REDUCED {job['id']}: {seconds:.3f}s; GPU {result['memory']['peak_cuda_allocated_bytes'] / 2**30:.3f}GiB",
            flush=True,
        )
        result["evaluations"] = quality(packed, output, "cuda:0", [0])
        result["status"] = "complete"
        write_json(destination / "result.json", result)
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
    records = []
    cells = []
    for job in manifest["jobs"]:
        path = folder / "jobs" / job["id"] / "result.json"
        records.append(
            json.loads(path.read_text()) if path.exists() else dict(job, status="pending")
        )
    for method in manifest["methods"]:
        for rate in manifest["rates"]:
            paired = {}
            for backend in ("author", "graphslim"):
                selected = [
                    r
                    for r in records
                    if (r["method"], r["rate"], r["backend"]) == (method, rate, backend)
                ]
                complete = len(selected) == len(manifest["seeds"]) and all(
                    r["status"] == "complete" and len(r.get("evaluations", [])) == 1
                    for r in selected
                )
                cell = dict(
                    method=method,
                    rate=rate,
                    backend=backend,
                    status="complete" if complete else "unavailable",
                    completed_seeds=sum(r["status"] == "complete" for r in selected),
                    expected_seeds=len(manifest["seeds"]),
                )
                if complete:
                    assert {r["seed"] for r in selected} == set(manifest["seeds"])
                    assert all(r["input_sha256"] == manifest["input_sha256"] for r in selected)
                    assert len({r["synthetic_nodes"] for r in selected}) == 1
                    assert all(
                        r["environment"]["gpu_uuid"] == manifest["gpu"]["uuid"] for r in selected
                    )
                    for field, values in [
                        ("reduction_seconds", [r["reduction_seconds"] for r in selected]),
                        (
                            "peak_cuda_allocated_bytes",
                            [r["memory"]["peak_cuda_allocated_bytes"] for r in selected],
                        ),
                        (
                            "peak_sampled_rss_bytes",
                            [r["memory"]["peak_sampled_rss_bytes"] for r in selected],
                        ),
                    ]:
                        cell[field] = dict(
                            values=values,
                            median=statistics.median(values),
                            minimum=min(values),
                            maximum=max(values),
                        )
                    cell.update(
                        synthetic_nodes=selected[0]["synthetic_nodes"],
                        accuracy_mean=100
                        * statistics.mean(r["evaluations"][0]["test_accuracy"] for r in selected),
                    )
                    paired[backend] = cell
                cells.append(cell)
            if len(paired) == 2:
                a, b = paired["author"], paired["graphslim"]
                assert a["synthetic_nodes"] == b["synthetic_nodes"]
                comparison = dict(
                    author_over_graphslim_time=a["reduction_seconds"]["median"]
                    / b["reduction_seconds"]["median"],
                    author_over_graphslim_gpu_memory=a["peak_cuda_allocated_bytes"]["median"]
                    / b["peak_cuda_allocated_bytes"]["median"],
                    relative_accuracy_shortfall_percent=100
                    * max(0, a["accuracy_mean"] - b["accuracy_mean"])
                    / a["accuracy_mean"],
                )
                for cell in paired.values():
                    cell.update(comparison)
    summary = dict(
        data=manifest["data"],
        protocol=manifest["protocol"],
        references=manifest["references"],
        manifest_sha256=sha(folder / "manifest.json"),
        cells=cells,
        records=records,
    )
    write_json(folder / "summary.json", summary)
    return summary


def run(options):
    import psutil

    folder = options.run_dir.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(folder / name) == expected, f"Frozen file changed: {name}"
    handles = []
    try:
        for path in [
            f"/tmp/ekpt-physical-gpu-{manifest['gpu']['index']}.lock",
            f"/tmp/graphslim-{manifest['gpu']['uuid']}.lock",
        ]:
            handle = open(path, "a+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        count = 0
        for job in manifest["jobs"]:
            target = folder / "jobs" / job["id"]
            if (target / "result.json").exists():
                continue
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
            )
            if manifest["gpu"]["uuid"] in apps:
                raise RuntimeError("Selected GPU is occupied; no existing job was interrupted")
            target.mkdir(parents=True, exist_ok=True)
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=manifest["gpu"]["uuid"],
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                MPLCONFIGDIR=str(folder / "mpl"),
            )
            print(f"START {job['id']}", flush=True)
            failure = None
            started = time.monotonic()
            with (target / "run.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(folder / "runner.py"),
                        "worker",
                        "--run-dir",
                        str(folder),
                        "--job",
                        job["id"],
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                monitored = psutil.Process(process.pid)
                while process.poll() is None:
                    if time.monotonic() - started > manifest["protocol"]["timeout_seconds"]:
                        failure = "timeout"
                    try:
                        rss = monitored.memory_info().rss
                        if rss > manifest["protocol"]["max_rss_gib"] * 2**30:
                            failure = "resource_limit"
                    except psutil.NoSuchProcess:
                        pass
                    if failure:
                        process.kill()
                        process.wait()
                        break
                    time.sleep(0.25)
            print(f"EXIT {job['id']}: {process.returncode}", flush=True)
            if failure:
                write_json(
                    target / "result.json",
                    dict(
                        job,
                        status=failure,
                        timeout_seconds=manifest["protocol"]["timeout_seconds"],
                        max_rss_gib=manifest["protocol"]["max_rss_gib"],
                    ),
                )
            summary = summarize(folder)
            done = sum(r["status"] == "complete" for r in summary["records"])
            print(f"PROGRESS {done}/{len(manifest['jobs'])}", flush=True)
            count += 1
            if options.job_limit and count >= options.job_limit:
                break
    finally:
        for handle in handles:
            handle.close()
    summarize(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "worker", "summarize", "freeze"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rates", nargs="+", choices=RATES, default=list(RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpu-index", type=int, default=2)
    parser.add_argument("--gpu-uuid", default="GPU-63b110a5-9b61-48c4-cc62-ec3b3010f952")
    parser.add_argument("--job")
    parser.add_argument("--job-limit", type=int)
    parser.add_argument(
        "--package-source",
        type=Path,
        help="measured GraphSlim directory extracted from the source archive",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "benchmark/results/arxiv_clustering_comparison_20261101.json",
    )
    options = parser.parse_args()
    if options.action == "freeze":
        freeze(options)
    elif options.action == "summarize":
        summarize(options.run_dir.resolve())
    else:
        globals()[options.action](options)


def freeze(options):
    import numpy as np
    import psutil
    import sklearn
    import torch

    folder = options.run_dir.resolve()
    summary = summarize(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert all(c["status"] == "complete" for c in summary["cells"]), (
        "Incomplete cells remain unavailable"
    )
    assert len(manifest["seeds"]) == 3 and manifest["methods"] == list(METHODS)
    for name, expected in manifest["files"].items():
        assert sha(folder / name) == expected, f"Frozen source changed: {name}"
    target = options.snapshot.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_name("arxiv_clustering_source_20261101.zip")
    # Original repositories remain at their pinned public URLs; do not republish
    # their source. Bundle the measured GraphSlim implementation and harness.
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted((folder / "source").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                bundle.write(path, str(path.relative_to(folder)))
        bundle.write(folder / "runner.py", "runner.py")
        bundle.write(folder / "manifest.json", "manifest.json")
    snapshot = dict(
        summary,
        exported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        execution_environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            psutil=psutil.__version__,
            cpu_model=next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            faiss_dependency_source="runs/implementation_comparison_deps_20261031/faiss_cpu-1.8.0.post1.dist-info",
        ),
        source_run=str(folder.relative_to(ROOT)),
        measurement_scope="The timed input is the common CPU graph after symmetrization and training-fitted standardization. Timing includes reducer-internal normalization, clustering, updates and transfers; the shared input preparation is excluded.",
        source_manifest=manifest,
        source_archive=dict(
            path=str(archive.relative_to(ROOT)),
            sha256=sha(archive),
            scope="Measured GraphSlim source and runner; author code is obtained from the pinned repositories in source_manifest.references.",
        ),
        result_sha256={
            r["id"]: sha(folder / "jobs" / r["id"] / "result.json") for r in summary["records"]
        },
    )
    write_json(target, snapshot)
    print(
        f"Frozen {len(snapshot['records'])} completed reductions in {target.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

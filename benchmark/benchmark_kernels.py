"""Paired GCGP SGNK microbenchmarks against pinned author mathematics.

Equal shapes/dtype/inputs and forward/gradient checks precede every timing row.
This is NOT end-to-end condensation, accuracy reproduction, or an all-method
speed claim. CUDA timing is synchronized; raw alternating repeats are retained.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time
import zipfile

import torch

from benchmark.rerun_nr import write_json, utcnow
from benchmark.verify_lightweight import load_definitions, REFERENCES
from graphslim.condensation.gcgp import neural_kernel, gp_predict

ROOT = Path(__file__).resolve().parents[1]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, count, device):
    synchronize(device)
    start = time.perf_counter()
    for _ in range(count):
        function()
    synchronize(device)
    return (time.perf_counter() - start) / count


def replay_cora(options, author_gp, device):
    """A controlled kernel replacement inside a shared Cora workflow, not the author's full repository."""
    from copy import deepcopy
    from graphslim.config import get_args
    from graphslim.dataset import get_dataset
    from graphslim.evaluation import Evaluator
    from graphslim.reduction.lightweight import normalized_adjacency, propagate
    from graphslim.utils import seed_everything
    from benchmark.rerun_nr import array_hash

    args = get_args(
        [
            "-D",
            "cora",
            "-M",
            "gcgp",
            "-R",
            "0.5",
            "-G",
            "-1",
            "--epochs",
            "200",
            "--eval_epochs",
            "300",
            "--hidden",
            "256",
            "--run_eval",
            "1",
            "--save_path",
            str(options.output.parent / "replay_scratch"),
        ]
    )
    args.device = str(device)
    args.eval_mode = "cross"
    seed_everything(1)
    graph = get_dataset("cora", args, str(ROOT / "data"))
    features = graph.feat_full.to(device)
    training = propagate(normalized_adjacency(graph.adj_full, device), features, 4)[
        graph.idx_train
    ].detach()
    dtype = getattr(torch, options.dtype)
    training = training.to(dtype)
    targets = torch.nn.functional.one_hot(graph.labels_train.to(device), graph.nclass).to(dtype)
    seed_everything(1)
    initialization = torch.Generator().manual_seed(1)
    initial_features = torch.rand(
        70, features.shape[1], generator=initialization, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    initial_labels = torch.rand(70, graph.nclass, generator=initialization, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    # A common CPU-generated initialization avoids CPU/CUDA RNG-stream differences.
    identity = torch.eye(70, device=device, dtype=dtype)
    study = {
        "scope": "Controlled SGNK/GP implementation replacement in shared package preprocessing and GCN evaluation; NOT an author-repository end-to-end reproduction",
        "dataset": "Cora public split",
        "reduction_seed": 1,
        "evaluation_seeds": [0, 1, 2],
        "coarse_nodes": 70,
        "reduction_steps": 200,
        "optimization_dtype": options.dtype,
        "downstream_dtype": "float32",
        "downstream_steps": 300,
        "hidden": 256,
        "lr": 0.01,
        "ridge": 0.5,
        "features": "raw BOW; four-hop normalized propagation",
        "timing_scope": "200 Adam updates, kernel/GP forward/backward and scalar loss records; the reference retains its extra training-accuracy computation. Preprocessing, parameter/optimizer construction and downstream fitting excluded.",
        "timed_optimization_runs_per_backend": 1,
        "optimizer_warmup_steps_per_backend": 2,
        "source_data_sha256": {
            name: array_hash(getattr(graph, name))
            for name in ("feat_full", "labels_train", "idx_train", "idx_val", "idx_test")
        },
        "initial_features_sha256": array_hash(initial_features),
        "initial_labels_sha256": array_hash(initial_labels),
        "backends": {},
    }
    final_parameters = {}

    def optimize(backend, epochs):
        feature = torch.nn.Parameter(initial_features.clone())
        label = torch.nn.Parameter(initial_labels.clone())
        optimizer = torch.optim.Adam([feature, label], lr=0.01)
        losses = []
        synchronize(device)
        started = time.perf_counter()
        for _ in range(epochs):
            if backend == "author_math":
                prediction, _ = author_gp(
                    training, feature, targets, label, None, identity, identity, learnA=False
                )
            else:
                prediction = gp_predict(
                    neural_kernel(training, feature), neural_kernel(feature, feature), label, 0.5
                )
            loss = torch.nn.functional.mse_loss(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        synchronize(device)
        elapsed = time.perf_counter() - started
        return feature.detach().clone(), label.detach().clone(), elapsed, losses

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        for backend in ("author_math", "package_math"):
            optimize(backend, 2)
        for backend in ("author_math", "package_math"):
            feature, label, elapsed, losses = optimize(backend, 200)
            final_parameters[backend] = (feature, label)
            study["backends"][backend] = {
                "optimization_seconds": elapsed,
                "loss_history": losses,
                "final_features_sha256": array_hash(feature),
                "final_labels_sha256": array_hash(label),
                "downstream": [],
            }
    finally:
        torch.set_default_dtype(previous_dtype)
    study["optimization_speedup"] = (
        study["backends"]["author_math"]["optimization_seconds"]
        / study["backends"]["package_math"]["optimization_seconds"]
    )
    study["final_features_max_absolute_difference"] = float(
        (final_parameters["author_math"][0] - final_parameters["package_math"][0]).abs().max()
    )
    study["final_labels_max_absolute_difference"] = float(
        (final_parameters["author_math"][1] - final_parameters["package_math"][1]).abs().max()
    )
    for backend, parameters in final_parameters.items():
        for seed in study["evaluation_seeds"]:
            evaluated = deepcopy(graph)
            evaluated.feat_syn, evaluated.labels_syn = (value.float() for value in parameters)
            evaluated.adj_syn = identity.float()
            local_args = deepcopy(args)
            local_args.seed = seed
            seed_everything(seed)
            validation, score = Evaluator(local_args).test(
                evaluated, "GCN", reduced=True, mode="cross", verbose=False
            )
            study["backends"][backend]["downstream"].append(
                {"seed": seed, "validation": float(validation), "test": float(score)}
            )
        scores = [row["test"] for row in study["backends"][backend]["downstream"]]
        study["backends"][backend]["test_mean"] = statistics.mean(scores)
        study["backends"][backend]["test_population_std"] = statistics.pstdev(scores)
    return study


def run(options):
    device = torch.device(options.device)
    torch.set_num_threads(options.threads)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    reference = options.references / "GCGP"
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCES["GCGP"][0]:
        raise RuntimeError("GCGP author source is not at the pinned commit")
    subprocess.run(
        ["git", "-C", str(reference), "diff", "--exit-code", "HEAD"],
        check=True,
        capture_output=True,
    )
    author_path = reference / "gcgp/sgnk.py"
    frozen_paths = [
        path
        for path in (ROOT / "graphslim").rglob("*")
        if path.suffix in (".py", ".json", ".npy", ".sh")
    ]
    frozen_paths += [
        Path(__file__),
        ROOT / "benchmark/__init__.py",
        ROOT / "benchmark/rerun_nr.py",
        ROOT / "benchmark/verify_lightweight.py",
        ROOT / "benchmark/reproduce.py",
        ROOT / "pyproject.toml",
        author_path,
        reference / "gcgp/gpr.py",
    ]
    frozen_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_paths}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source_archive = options.output.with_suffix(".source.zip")
    with zipfile.ZipFile(source_archive, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(frozen_paths):
            name = (
                "author/" + path.name
                if path in (author_path, reference / "gcgp/gpr.py")
                else str(path.relative_to(ROOT))
            )
            archive.write(path, name)
    author = load_definitions(author_path, {"SimplifyingGraphNeuralKernel"})[
        "SimplifyingGraphNeuralKernel"
    ](L=2)
    gp = load_definitions(reference / "gcgp/gpr.py", {"GaussianProcessRegression"})[
        "GaussianProcessRegression"
    ](author.nodes_gram, 0.5)
    report = {
        "created_utc": utcnow(),
        "purpose": __doc__,
        "device": str(device),
        "environment": {
            p: importlib.metadata.version(p) for p in ("torch", "torch-geometric", "numpy")
        },
        "torch_cuda_runtime": torch.version.cuda,
        "threads": options.threads,
        "iterations_per_repeat": options.iterations,
        "repeats": options.repeats,
        "warmups": options.warmups,
        "reference_commit": commit,
        "source_archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        "reference_url": REFERENCES["GCGP"][1] + "/blob/" + commit + "/gcgp/sgnk.py",
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("benchmark_script", Path(__file__)),
                ("package_kernel", ROOT / "graphslim/condensation/gcgp.py"),
                ("author_kernel", author_path),
            )
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rows": [],
    }
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        report["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    if device.type == "cuda":
        report["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    shapes = [(140, 70, 1433), (1024, 256, 128), (4096, 256, 128)]
    for queries, synthetic, dimensions in shapes:
        generator = torch.Generator().manual_seed(36)
        dtype = getattr(torch, options.dtype)
        query = (0.1 * torch.randn(queries, dimensions, generator=generator, dtype=dtype)).to(
            device
        )
        values = (
            (0.1 * torch.randn(synthetic, dimensions, generator=generator, dtype=dtype))
            .to(device)
            .requires_grad_()
        )
        functions = {
            "author": lambda: author.nodes_gram(query, values, None, None),
            "package": lambda: neural_kernel(query, values),
        }
        outputs = {name: function() for name, function in functions.items()}
        torch.testing.assert_close(outputs["package"], outputs["author"], rtol=1e-5, atol=1e-5)
        gradients = {
            name: torch.autograd.grad(value.square().mean(), values)[0]
            for name, value in outputs.items()
        }
        torch.testing.assert_close(gradients["package"], gradients["author"], rtol=1e-4, atol=1e-5)
        onehot = torch.nn.functional.one_hot(torch.arange(queries, device=device) % 7, 7).float()
        targets = torch.rand(synthetic, 7, generator=generator, dtype=dtype).to(device)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            reference_prediction, _ = gp(
                query, values, onehot, targets, None, None, None, learnA=False
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        prediction = gp_predict(
            neural_kernel(query, values), neural_kernel(values, values), targets, 0.5
        )
        torch.testing.assert_close(prediction, reference_prediction, rtol=1e-4, atol=1e-5)
        row = {
            "query_nodes": queries,
            "synthetic_nodes": synthetic,
            "features": dimensions,
            "dtype": options.dtype,
            "forward_max_absolute_error": float(
                (outputs["package"] - outputs["author"]).detach().abs().max()
            ),
            "gradient_max_absolute_error": float(
                (gradients["package"] - gradients["author"]).abs().max()
            ),
            "gp_prediction_max_absolute_error": float(
                (prediction - reference_prediction).detach().abs().max()
            ),
            "avoided_full_self_gram_bytes": query.element_size() * (queries**2 + synthetic**2),
            "timings": {},
        }
        del outputs, gradients, prediction, reference_prediction
        gc.collect()
        for stage in ("forward", "forward_backward"):

            def operation(name):
                if stage == "forward":
                    with torch.no_grad():
                        return functions[name]()
                output = functions[name]()
                return torch.autograd.grad(output.square().mean(), values)[0]

            times = {name: [] for name in functions}
            for name in functions:
                for _ in range(options.warmups):
                    operation(name)
            for repeat in range(options.repeats):
                order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                for name in order:
                    times[name].append(timed(lambda: operation(name), options.iterations, device))
            result = {
                name: {"median_seconds": statistics.median(samples), "raw_seconds": samples}
                for name, samples in times.items()
            }
            result["speedup"] = (
                result["author"]["median_seconds"] / result["package"]["median_seconds"]
            )
            if device.type == "cuda":
                for name in functions:
                    gc.collect()
                    torch.cuda.empty_cache()
                    synchronize(device)
                    baseline = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    operation(name)
                    synchronize(device)
                    result[name]["additional_peak_cuda_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated(device) - baseline
                    )
            row["timings"][stage] = result
        report["rows"].append(row)
        write_json(options.output, report)
        print(
            f"{device} SGNK ({queries},{synthetic},{dimensions}): forward {row['timings']['forward']['speedup']:.2f}x; forward/backward {row['timings']['forward_backward']['speedup']:.2f}x",
            flush=True,
        )
    report["completed_utc"] = utcnow()
    if options.cora_replay:
        report["cora_replay"] = replay_cora(options, gp, device)
        report["completed_utc"] = utcnow()
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != expected
        for path, expected in frozen_hashes.items()
    ):
        raise RuntimeError(
            "Scientific source changed during the benchmark; use a new output and rerun."
        )
    write_json(options.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--references", type=Path, default=ROOT / "runs/reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--cora-replay",
        action="store_true",
        help="fixed 200-step Cora kernel replacement plus three 300-step GCN evaluations per backend",
    )
    options = parser.parse_args(argv)
    if min(options.threads, options.iterations, options.repeats) < 1 or options.warmups < 0:
        parser.error("positive threads/iterations/repeats and nonnegative warmups required")
    run(options)


if __name__ == "__main__":
    main()

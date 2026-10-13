"""Real lightweight reduction pairs with/without the optional CUDA backend.

Toy reductions are integration tests. Cora is a matched package comparison,
not an author-repository reproduction or a new full benchmark table.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import torch

from benchmark.rerun_nr import source_digest, array_hash
from benchmark.verify_cuda_graph import toy_graph
from graphslim.acceleration import acceleration_records
from graphslim.config import get_args
from graphslim.dataset import get_dataset
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=["toy", "cora"], default="toy")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gcgp", "clustgdd", "cgc", "nope"],
        default=["gcgp", "clustgdd"],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError("Use a distinct frozen output path")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    digest = source_digest()
    record = {
        "scope": "Actual package reductions paired by source, input, seed, arithmetic and budget. Toy is integration only; Cora is not author-repository reproduction. No downstream test selection.",
        "dataset": options.dataset,
        "device": options.device,
        "steps": options.steps,
        "source_sha256": digest,
        "status": "running",
        "cases": [],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": True,
    }
    for method in options.methods:
        arithmetic_choices = ["package", "released"] if method == "gcgp" else ["package"]
        for arithmetic in arithmetic_choices:
            for seed in options.seeds:
                args = get_args(
                    [
                        "-D",
                        "cora",
                        "-M",
                        method,
                        "-G",
                        "-1",
                        "-R",
                        ".5",
                        "--epochs",
                        str(options.steps),
                        "--run_eval",
                        "1",
                        "--save_path",
                        str(options.output.parent / "scratch"),
                    ]
                )
                args.device, args.seed = options.device, seed
                args.gcgp_arithmetic = arithmetic
                args.cuda_graph_min_steps = 1
                args._cuda_graph_trace = True
                if options.dataset == "toy":
                    args.clustgdd_hidden = 16
                seed_everything(seed)
                graph = (
                    toy_graph()
                    if options.dataset == "toy"
                    else get_dataset(
                        "cora", args, str(Path(__file__).resolve().parents[1] / "data")
                    )
                )
                case = {"method": method, "arithmetic": arithmetic, "seed": seed}
                tensors = {}
                try:
                    for enabled in (False, True):
                        local_args = deepcopy(args)
                        local_args.cuda_graph = enabled
                        data = deepcopy(graph)
                        seed_everything(seed)
                        reducer = create_reducer(
                            method, setting="trans", data=data, args=local_args
                        )
                        if options.device.startswith("cuda"):
                            torch.cuda.synchronize(options.device)
                        started = time.perf_counter()
                        data = reducer.reduce(data, verbose=False, save=False)
                        if options.device.startswith("cuda"):
                            torch.cuda.synchronize(options.device)
                        elapsed = time.perf_counter() - started
                        values = {
                            name: getattr(data, name).detach().cpu().clone()
                            for name in ("feat_syn", "adj_syn", "labels_syn")
                        }
                        label = "cuda_graph" if enabled else "eager"
                        tensors[label] = values
                        case[label] = {
                            "reduction_seconds_including_setup": elapsed,
                            "hashes": {name: array_hash(value) for name, value in values.items()},
                            "metadata": data.reduction_metadata,
                            "stages": acceleration_records(local_args),
                            "GP_loss_trace": getattr(local_args, "_reduction_loss_trace", None),
                        }
                    case["tensors_equal"] = all(
                        torch.equal(value, tensors["cuda_graph"][name])
                        for name, value in tensors["eager"].items()
                    )
                    case["losses_equal"] = case["eager"]["GP_loss_trace"] == case["cuda_graph"][
                        "GP_loss_trace"
                    ] and [r.get("loss_trace") for r in case["eager"]["stages"]] == [
                        r.get("loss_trace")
                        for r in case["cuda_graph"]["stages"]
                        if "loss_trace" in r
                    ]
                    # A partial kernel record has no scalar loss trace; the GP
                    # reducer records its full real loss history separately.
                    if method == "gcgp":
                        case["losses_equal"] = (
                            case["eager"]["GP_loss_trace"] == case["cuda_graph"]["GP_loss_trace"]
                        )
                    case["speedup_including_setup"] = (
                        case["eager"]["reduction_seconds_including_setup"]
                        / case["cuda_graph"]["reduction_seconds_including_setup"]
                    )
                    case["status"] = (
                        "pass" if case["tensors_equal"] and case["losses_equal"] else "difference"
                    )
                except Exception as error:
                    case.update(status="error", error=f"{type(error).__name__}: {error}")
                record["cases"].append(case)
                options.output.write_text(json.dumps(record, indent=2))
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in case.items()
                            if key not in ("eager", "cuda_graph")
                        }
                    ),
                    flush=True,
                )
    record["status"] = (
        "invalid_source_changed"
        if source_digest() != digest
        else "complete"
        if all(c["status"] == "pass" for c in record["cases"])
        else "failed"
    )
    options.output.write_text(json.dumps(record, indent=2))
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

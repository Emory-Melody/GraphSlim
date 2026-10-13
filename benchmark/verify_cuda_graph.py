"""Matched all-method-tag/downstream integration checks, not reducer reproductions.

Use an explicitly leased CUDA device. Clean/toy noise cases exercise the actual
package fitting loop; checkpoint weights, validation/test and loss histories are
compared. Setup/verification cost is included in whole-fit timings.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from benchmark.rerun_nr import array_hash, source_digest
from graphslim.acceleration import acceleration_records, method_acceleration_support
from graphslim.config import get_args
from graphslim.dataset.noise import NOISE_TYPES, perturb_graph
from graphslim.models import DOWNSTREAM_MODELS, get_model_class
from graphslim.reduction import list_methods
from graphslim.utils import seed_everything


def toy_graph():
    generator = torch.Generator().manual_seed(91)
    nodes = 48
    labels = torch.arange(nodes) % 2
    features = torch.rand(nodes, 16, generator=generator) + labels[:, None].float()
    edges = [(i, (i + offset) % nodes) for i in range(nodes) for offset in (1, 2, 4)]
    adj = sp.csr_matrix((np.ones(len(edges)), np.asarray(edges).T), shape=(nodes, nodes))
    adj = adj.maximum(adj.T).astype(np.float32)
    data = SimpleNamespace(
        feat_full=features,
        labels_full=labels,
        adj_full=adj,
        idx_train=torch.arange(24),
        idx_val=torch.arange(24, 36),
        idx_test=torch.arange(36, 48),
        nclass=2,
        num_nodes=nodes,
        x=features,
        y=labels,
        edge_index=torch.tensor(np.stack(adj.nonzero())),
    )
    for split in ("train", "val", "test"):
        index = getattr(data, f"idx_{split}")
        setattr(data, f"feat_{split}", features[index])
        setattr(data, f"labels_{split}", labels[index])
        setattr(data, f"adj_{split}", adj[index.numpy()][:, index.numpy()])
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[index] = True
        setattr(data, f"{split}_mask", mask)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--methods", nargs="+", choices=list_methods(), default=list(list_methods())
    )
    parser.add_argument(
        "--models", nargs="+", choices=DOWNSTREAM_MODELS, default=list(DOWNSTREAM_MODELS)
    )
    parser.add_argument("--noise", nargs="+", choices=NOISE_TYPES, default=["clean"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic Torch algorithms on BOTH sides of this parity audit",
    )
    parser.add_argument("--setting", choices=["trans", "ind"], default="trans")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError("Use a new output path for each frozen source/protocol")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if options.deterministic:
        # Required by CuBLAS for deterministic matmul on CUDA >=10.2. Set this
        # before creating the first CUDA tensor/context in this audit process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(options.deterministic)
    digest = source_digest()
    models = {name: get_model_class(name) for name in options.models}
    result = {
        "scope": "Synthetic-graph method-tag x downstream-model integration, NOT actual reductions or scientific dataset accuracy; timings include graph setup/verification, validation and per-step diagnostic loss recording.",
        "source_sha256": digest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": options.device,
        "steps": options.steps,
        "setting": options.setting,
        "deterministic_algorithms": options.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu": torch.cuda.get_device_name(options.device)
        if options.device.startswith("cuda")
        else None,
        "policy": method_acceleration_support(),
        "cases": [],
        "status": "running",
        "expected_cases": len(options.methods) * len(options.models) * len(options.noise),
    }
    graph = toy_graph()

    def save():
        options.output.write_text(json.dumps(result, indent=2))

    save()
    for method in options.methods:
        for noise in options.noise:
            data = perturb_graph(graph, noise, 0.2, 31, setting=options.setting)
            # An explicit common synthetic graph, not a pretended reducer output.
            data.feat_syn = data.feat_train[:12].clone()
            data.adj_syn = torch.as_tensor(data.adj_train[:12, :12].toarray())
            data.labels_syn = data.labels_train[:12].clone()
            if method == "gcgp":
                data.labels_syn = torch.nn.functional.one_hot(data.labels_syn, 2).float()
            for name in options.models:
                args = get_args(
                    [
                        "-M",
                        method,
                        "-G",
                        "-1",
                        "--epochs",
                        "2",
                        "--eval_epochs",
                        str(options.steps),
                        "--run_eval",
                        "1",
                        "--hidden",
                        "16",
                        "--save_path",
                        str(options.output.parent / "scratch"),
                    ]
                )
                args.device, args.setting = options.device, options.setting
                args.cuda_graph_min_steps = (
                    1  # Short diagnostics intentionally bypass amortization policy.
                )
                args._cuda_graph_trace = True
                args.eval_mode = "cross"
                args.trans_num_layers = 1
                case_data = deepcopy(data)
                if method == "geom" and args.soft_label:
                    # GEOM's default KL(log_target=True) requires log-probability
                    # synthetic targets, not the hard labels used by other tags.
                    probability = torch.nn.functional.one_hot(data.labels_train[:12], 2).float()
                    case_data.labels_syn = (0.9 * probability + 0.05).log()
                case = {"method": method, "model": name, "noise": noise}
                snapshots = {}
                try:
                    for enabled in (False, True):
                        local_args = deepcopy(args)
                        local_args.cuda_graph = enabled
                        seed_everything(31)
                        model = models[name](16, 16, 2, local_args, mode="eval").to(options.device)
                        if options.device.startswith("cuda"):
                            torch.cuda.synchronize(options.device)
                        started = time.perf_counter()
                        validation = model.fit_with_val(
                            deepcopy(case_data),
                            train_iters=options.steps,
                            setting=options.setting,
                            reduced=True,
                            verbose=False,
                        )
                        if options.device.startswith("cuda"):
                            torch.cuda.synchronize(options.device)
                        elapsed = time.perf_counter() - started
                        score = model.test(case_data, setting=options.setting, verbose=False)
                        weights = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        records = acceleration_records(local_args)
                        label = "cuda_graph" if enabled else "eager"
                        snapshots[label] = weights
                        case[label] = {
                            "fit_seconds": elapsed,
                            "validation": float(validation),
                            "test": float(score),
                            "parameter_hashes": {
                                key: array_hash(value) for key, value in weights.items()
                            },
                            "stages": records,
                        }
                    case["weights_equal"] = all(
                        torch.equal(value, snapshots["cuda_graph"][key])
                        for key, value in snapshots["eager"].items()
                    )
                    case["losses_equal"] = [
                        r.get("loss_trace") for r in case["eager"]["stages"]
                    ] == [r.get("loss_trace") for r in case["cuda_graph"]["stages"]]
                    case["scores_equal"] = all(
                        case["eager"][key] == case["cuda_graph"][key]
                        for key in ("validation", "test")
                    )
                    case["fit_speedup_including_setup"] = (
                        case["eager"]["fit_seconds"] / case["cuda_graph"]["fit_seconds"]
                    )
                    case["status"] = (
                        "pass"
                        if all(
                            case[key] for key in ("weights_equal", "losses_equal", "scores_equal")
                        )
                        else "difference"
                    )
                except Exception as error:
                    case.update(status="error", error=f"{type(error).__name__}: {error}")
                result["cases"].append(case)
                save()
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
    if source_digest() != digest:
        result["status"] = "invalid_source_changed"
    else:
        result["status"] = (
            "complete" if all(row["status"] == "pass" for row in result["cases"]) else "failed"
        )
    save()
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

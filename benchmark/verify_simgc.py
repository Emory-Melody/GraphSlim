"""Validate a fresh SimGC citation reduction without computing test metrics.

This is a one-seed package integration check, not a benchmark result. Run in a
new directory; an explicitly selected physical GPU must be idle and leased.
"""

import argparse
import ast
from copy import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from benchmark import paper_hpo as hpo
from benchmark import rerun_nr as nr


def check_author_inputs(reference, graph, args):
    """Execute the pinned converter, including its raw-feature access semantics."""
    import numpy as np
    import scipy.sparse as sp
    import torch
    from deeprobust.graph.data import Dataset
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    tree = ast.parse((reference / "utils.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"Pyg2Dpr", "mask_to_index"}
    ]
    if {node.name for node in nodes} != {"Pyg2Dpr", "mask_to_index"}:
        raise RuntimeError("Pinned author input converter is missing")
    namespace = {"Dataset": Dataset, "np": np, "sp": sp}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(reference / "utils.py"), "exec"),
        namespace,
    )
    dataset = Planetoid(args.load_path, args.dataset, transform=NormalizeFeatures())
    author = namespace["Pyg2Dpr"](dataset)
    features = torch.as_tensor(author.features)
    matches = {
        "features_equal": torch.equal(features, graph.feat_full.cpu()),
        "adjacency_equal": (author.adj != graph.adj_full).nnz == 0,
    }
    for split in ("train", "val", "test"):
        matches[f"{split}_indices_equal"] = np.array_equal(
            getattr(author, f"idx_{split}"), getattr(graph, f"idx_{split}")
        )
    if not all(matches.values()):
        raise RuntimeError(f"Author input parity failed: {matches}")
    return {
        **matches,
        "author_features_sha256": nr.array_hash(features),
        "author_equals_raw_features": torch.equal(features, dataset._data.x),
        "author_equals_transformed_features": torch.equal(features, dataset[0].x),
        "mean_feature_row_sum": float(features.sum(dim=1).mean()),
        "scope": "input and split metadata parity; no test scoring",
    }


def check_author_models(reference, args):
    """Compare supplied-weight forward/input/edge gradients to pinned author code."""
    import torch
    from torch_sparse import SparseTensor
    from graphslim.models import GCN
    from graphslim.condensation.simgc import SimGCTeacher

    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "1b3d1d59d9afaf2c9c0c7b652fb183db2a5ffa01"
    if commit != expected:
        raise RuntimeError("Reference does not match the pinned author commit")
    if subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Author reference has local modifications")
    sys.path.insert(0, str(reference))
    from models.basicgnn import GCN as AuthorGCN
    from models.sgc_multi import SGC_Multi as AuthorTeacher

    cpu_args = copy(args)
    cpu_args.device, cpu_args.hidden = "cpu", 4
    cpu_args.ntrans, cpu_args.nlayers = 2, 2
    cpu_args.activation, cpu_args.dropout, cpu_args.with_bn = None, 0, False
    pairs = (
        (
            "gcn",
            GCN(3, 4, 2, cpu_args, mode="cross"),
            AuthorGCN(nfeat=3, nhid=4, nclass=2, nlayers=2, dropout=0, norm=None, act=None),
        ),
        (
            "teacher",
            SimGCTeacher(3, 4, 2, cpu_args),
            AuthorTeacher(nfeat=3, nhid=4, nclass=2, K=2, nlayers=2, dropout=0, norm=None),
        ),
    )
    torch.manual_seed(17)
    features = torch.randn(3, 3)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 1, 2]])
    weights = torch.tensor([0.8, 0.2, 0.2, 0.6, 0.3, 0.7])
    reports = {}
    for name, port, author in pairs:
        port.eval()
        author.eval()
        with torch.no_grad():
            for p, a in zip(port.layers, author.convs if name == "gcn" else author.mlp.lins):
                p.weight.copy_(a.lin.weight.T if name == "gcn" else a.weight)
                p.bias.copy_(a.bias)
        x_port, x_author = features.clone().requires_grad_(), features.clone().requires_grad_()
        w_port, w_author = weights.clone().requires_grad_(), weights.clone().requires_grad_()
        adjacency = SparseTensor(row=edges[1], col=edges[0], value=w_port, sparse_sizes=(3, 3))
        output_port = port(x_port, adjacency)
        output_author = author(x_author, edges, edge_weight=w_author)
        grad_port = torch.autograd.grad(output_port[:, 0].sum(), (x_port, w_port))
        grad_author = torch.autograd.grad(output_author[:, 0].sum(), (x_author, w_author))
        errors = {
            "forward_max_abs": float((output_port - output_author).detach().abs().max()),
            "feature_gradient_max_abs": float((grad_port[0] - grad_author[0]).abs().max()),
            "edge_gradient_max_abs": float((grad_port[1] - grad_author[1]).abs().max()),
        }
        if not all(value < 1e-5 for value in errors.values()):
            raise RuntimeError(f"Author {name} parity failed: {errors}")
        reports[name] = errors
    return {
        "commit": commit,
        "models": reports,
        "scope": "supplied-weight operator parity, not native end-to-end reproduction",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, help="explicit physical GPU index")
    parser.add_argument("--dataset", choices=("cora", "citeseer"), default="cora")
    parser.add_argument("--rate", default="0.5")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--reference", type=Path, default=nr.ROOT / "runs/reference/SimGC")
    options = parser.parse_args(argv)
    folder = options.run_dir.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(
            "Use a new empty run directory; prior verification artifacts are immutable"
        )
    gpu = next(
        (gpu for gpu in nr.gpu_inventory() if gpu["index"] == options.gpu and gpu["idle"]), None
    )
    if gpu is None:
        raise RuntimeError("The requested GPU is not idle")
    handles = nr.acquire_gpu(gpu)
    if not handles:
        raise RuntimeError("The requested GPU lease is unavailable")
    started = time.monotonic()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        import torch
        from graphslim.config import get_args
        from graphslim.dataset import get_dataset
        from graphslim.models import GCN
        from graphslim.reduction import create_reducer
        from graphslim.utils import seed_everything

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Verification needs exactly one UUID-masked GPU")
        torch.set_num_threads(2)
        files = hpo.scientific_files() + [Path(__file__).resolve()]
        source = nr.source_digest(files)
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(nr.ROOT))
        nr.write_json(
            folder / "manifest.json",
            {
                "created_utc": nr.utcnow(),
                "source_sha256": source,
                "gpu": gpu,
                "dataset": options.dataset,
                "rate": options.rate,
                "seed": options.seed,
                "protocol": "one fresh reduction; author citation profiles; validation only; no test scoring; not a table result",
            },
        )
        args = get_args(
            [
                "-M",
                "simgc",
                "-D",
                options.dataset,
                "-R",
                options.rate,
                "-S",
                str(options.seed),
                "-G",
                "0",
                "--load_path",
                str(nr.ROOT / "data"),
                "--save_path",
                str(folder / "checkpoints"),
            ]
        )
        args.validation_only = True
        nr.write_json(
            folder / "config.json",
            {key: nr.serializable(value) for key, value in vars(args).items()},
        )
        nr.write_json(
            folder / "author_model_parity.json",
            check_author_models(options.reference.resolve(), args),
        )
        print("Pinned author GCN/SGC_Multi forward and gradient parity passed", flush=True)
        seed_everything(nr.DATA_SEED)
        with nr.lock_file(nr.ROOT / "data" / f".{options.dataset}.prepare.lock"):
            graph = get_dataset(options.dataset, args, args.load_path)
        nr.write_json(folder / "dataset.json", hpo.dataset_record(graph))
        nr.write_json(
            folder / "author_input_parity.json",
            check_author_inputs(options.reference.resolve(), graph, args),
        )
        print("Pinned author features, adjacency and split parity passed", flush=True)
        seed_everything(options.seed)
        nr.write_json(folder / "progress.json", {"stage": "reducing", "utc": nr.utcnow()})
        create_reducer("simgc", setting=args.setting, data=graph, args=args).reduce(
            graph, verbose=False
        )
        hpo.load_checkpoint(graph, args)
        nr.write_json(
            folder / "progress.json", {"stage": "validating_saved_graph", "utc": nr.utcnow()}
        )
        seed_everything(0)
        model = GCN(
            graph.feat_syn.shape[1], args.hidden, graph.nclass, args, mode=args.eval_mode
        ).to(args.device)
        validation = model.fit_with_val(
            graph, train_iters=args.eval_epochs, reduced=True, setting=args.setting
        )
        if nr.source_digest(files) != source:
            raise RuntimeError("Scientific source changed during verification")
        if not torch.isfinite(torch.tensor(validation)) or not 0 <= validation <= 1:
            raise RuntimeError("Non-finite validation score")
        result = {
            "completed_utc": nr.utcnow(),
            "source_sha256": source,
            "validation": validation,
            "evaluation_seed": 0,
            "reduction_seed": options.seed,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_sha256": {
                str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in hpo.checkpoint_paths(args)
            },
            "scope": "one-seed validation-only integration check; no test metric; not a full reproduction result",
        }
        nr.write_json(folder / "result.json", result)
        print(f"Validation-only check complete: {validation:.4f}", flush=True)
    except Exception as exc:
        nr.write_json(folder / "error.json", {"utc": nr.utcnow(), "error": repr(exc)})
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()

"""Deterministic, non-mutating graph corruptions for train-time and inference tests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from graphslim.reduction.lightweight import as_csr

NOISE_TYPES = (
    "clean",
    "random_adj",
    "edge_drop",
    "random_feat",
    "feature_mask",
    "gaussian",
    "label_noise",
)


def graph_fingerprint(data, setting):
    """Cache identity includes graph contents and split, not only dataset name."""
    digest = hashlib.sha256()
    prefix = "full" if setting == "trans" else "train"
    features = torch.as_tensor(getattr(data, f"feat_{prefix}")).detach().cpu().numpy()
    adjacency = as_csr(getattr(data, f"adj_{prefix}"))
    for value in (
        features,
        adjacency.indptr,
        adjacency.indices,
        adjacency.data,
        torch.as_tensor(data.idx_train).cpu().numpy(),
        torch.as_tensor(data.idx_val).cpu().numpy(),
        torch.as_tensor(data.idx_test).cpu().numpy(),
    ):
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _structural_noise(adjacency, kind, rate, rng):
    adjacency = as_csr(adjacency)
    n = len(adjacency.indptr) - 1
    adjacency = adjacency.maximum(adjacency.T)
    row, col = sp.triu(adjacency, k=1).nonzero()
    edges = set(zip(row.tolist(), col.tolist()))
    count = int(len(edges) * rate)
    modified = adjacency.tolil()
    if kind == "edge_drop":
        candidates = sorted(edges)
        for index in rng.permutation(len(candidates))[:count]:
            u, v = candidates[index]
            modified[u, v] = modified[v, u] = 0
    else:
        capacity = n * (n - 1) // 2 - len(edges)
        count = min(count, capacity)
        selected = set()
        if capacity < n * (n - 1) // 4:
            candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edges]
            selected = {candidates[i] for i in rng.permutation(len(candidates))[:count]}
        else:
            while len(selected) < count:
                u, v = rng.integers(n, size=2)
                if u != v:
                    pair = (int(min(u, v)), int(max(u, v)))
                    if pair not in edges:
                        selected.add(pair)
        for u, v in selected:
            modified[u, v] = modified[v, u] = 1
    modified = modified.tocsr()
    modified.eliminate_zeros()
    return modified


def perturb_graph(data, kind, rate=0.1, seed=0, *, setting="trans", stage="training"):
    """Return an independent graph. Held-out labels are never altered.

    Transductive feature/edge perturbations affect the visible full graph.
    Inductive training noise affects only the training graph; inference noise
    affects only the test graph. Gaussian sigma is rate * training-feature std.
    random_feat masks columns, feature_mask masks individual entries.
    """
    if kind not in NOISE_TYPES or not 0 <= rate <= 1:
        raise ValueError(f"Unknown noise or rate outside [0,1]: {kind}, {rate}")
    if setting not in ("trans", "ind") or stage not in ("training", "inference"):
        raise ValueError("Expected trans/ind and training/inference")
    if kind == "label_noise" and stage != "training":
        raise ValueError("Inference noise must not change ground-truth labels")
    result = deepcopy(data)
    prefix = "full" if setting == "trans" else ("train" if stage == "training" else "test")
    rng = np.random.default_rng(seed)
    if kind in ("random_adj", "edge_drop") and rate:
        setattr(
            result,
            f"adj_{prefix}",
            _structural_noise(getattr(result, f"adj_{prefix}"), kind, rate, rng),
        )
        if setting == "trans":
            for split in ("train", "val", "test"):
                indices = torch.as_tensor(getattr(result, f"idx_{split}")).cpu().numpy()
                setattr(result, f"adj_{split}", result.adj_full[indices][:, indices])
            result.edge_index = torch.tensor(np.stack(result.adj_full.nonzero()), dtype=torch.long)
        # Cached sampling structures must not refer to the clean adjacency.
        for name in ("sparse_adj", "samplers", "class_dict", "class_dict2"):
            if hasattr(result, name):
                setattr(result, name, None)
    elif kind in ("random_feat", "feature_mask", "gaussian") and rate:
        features = torch.as_tensor(getattr(result, f"feat_{prefix}")).clone()
        if kind == "random_feat":
            columns = rng.permutation(features.shape[1])[: int(rate * features.shape[1])]
            features[:, columns] = 0
        elif kind == "feature_mask":
            mask = torch.as_tensor(rng.random(features.shape) < rate, device=features.device)
            features[mask] = 0
        else:
            reference = torch.as_tensor(data.feat_train).to(features.device)
            scale = reference.std(0, unbiased=False)
            noise = torch.as_tensor(
                rng.standard_normal(features.shape), dtype=features.dtype, device=features.device
            )
            features = features + rate * scale * noise
        setattr(result, f"feat_{prefix}", features)
        if setting == "ind":
            indices = torch.as_tensor(
                getattr(result, f"idx_{prefix}"), device=result.feat_full.device
            )
            result.feat_full = result.feat_full.clone()
            result.feat_full[indices] = features.to(result.feat_full.device)
        else:
            for split in ("train", "val", "test"):
                setattr(result, f"feat_{split}", features[getattr(result, f"idx_{split}")])
        result.x = result.feat_full
    elif kind == "label_noise" and rate:
        labels = torch.as_tensor(result.labels_train).clone()
        if result.nclass < 2:
            raise ValueError("Label noise requires at least two classes")
        chosen = rng.permutation(len(labels))[: int(rate * len(labels))]
        offset = torch.as_tensor(
            rng.integers(1, result.nclass, size=len(chosen)), device=labels.device
        )
        labels[chosen] = (labels[chosen] + offset) % result.nclass
        result.labels_train = labels
        result.labels_full = result.labels_full.clone()
        result.labels_full[result.idx_train] = labels.to(result.labels_full.device)
        result.y = result.labels_full
    if kind != "clean" and rate:
        # Class samplers can cache labels as well as graph structure.
        for name in ("sparse_adj", "samplers", "class_dict", "class_dict2"):
            if hasattr(result, name):
                setattr(result, name, None)
    result.noise_metadata = {
        "kind": kind,
        "rate": rate,
        "seed": seed,
        "setting": setting,
        "stage": stage,
    }
    return result


def cached_noise(data, args):
    """Validated tensor/CSR cache for the CLI's inexpensive training noise."""
    signature = {
        "kind": args.attack,
        "rate": args.ptb_r,
        "seed": args.seed,
        "setting": args.setting,
        "stage": "training",
    }
    if getattr(data, "noise_metadata", None) == signature:
        return data
    source = graph_fingerprint(data, args.setting)
    # Training labels also belong to the identity for label corruption.
    labels_hash = hashlib.sha256(
        torch.as_tensor(data.labels_train).cpu().numpy().tobytes()
    ).hexdigest()[:12]
    folder = Path(args.save_path) / "corrupt_graph" / args.attack
    folder.mkdir(parents=True, exist_ok=True)
    prefix = (
        folder
        / f"{args.dataset}_{args.ptb_r}_{args.seed}_{args.setting}_{source[:16]}_{labels_hash}"
    )
    metadata = Path(str(prefix) + ".json")
    payload = Path(str(prefix) + ".pt")
    expected = {**signature, "source_sha256": source, "cache_version": 2}
    if payload.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == expected:
        result = deepcopy(data)
        values = torch.load(payload, map_location="cpu", weights_only=True)
        required = {"features", "labels", "edge_indices", "edge_values", "shape"}
        if set(values) != required:
            raise ValueError("Invalid noise cache fields")
        prefix_name = "full" if args.setting == "trans" else "train"
        reference = torch.as_tensor(getattr(data, f"feat_{prefix_name}"))
        if values["features"].shape != reference.shape:
            raise ValueError("Noise cache feature shape mismatch")
        setattr(result, f"feat_{prefix_name}", values["features"].to(reference.device))
        if tuple(values["shape"]) != tuple(getattr(data, f"adj_{prefix_name}").shape):
            raise ValueError("Noise cache adjacency shape mismatch")
        graph = sp.csr_matrix(
            (values["edge_values"].numpy(), values["edge_indices"].numpy()),
            shape=tuple(values["shape"]),
        )
        setattr(result, f"adj_{prefix_name}", graph)
        result.labels_train = values["labels"].to(data.labels_train.device)
        result.labels_full = result.labels_full.clone()
        result.labels_full[result.idx_train] = result.labels_train.to(result.labels_full.device)
        if args.setting == "trans":
            for split in ("train", "val", "test"):
                indices = getattr(result, f"idx_{split}")
                setattr(result, f"feat_{split}", result.feat_full[indices])
                idx_numpy = torch.as_tensor(indices).cpu().numpy()
                setattr(result, f"adj_{split}", graph[idx_numpy][:, idx_numpy])
            result.edge_index = values["edge_indices"].long()
        else:
            result.feat_full = result.feat_full.clone()
            result.feat_full[result.idx_train] = result.feat_train.to(result.feat_full.device)
        result.x, result.y = result.feat_full, result.labels_full
        for name in ("sparse_adj", "samplers", "class_dict", "class_dict2"):
            if hasattr(result, name):
                setattr(result, name, None)
        result.noise_metadata = signature
        return result
    result = perturb_graph(data, args.attack, args.ptb_r, args.seed, setting=args.setting)
    prefix_name = "full" if args.setting == "trans" else "train"
    graph = as_csr(getattr(result, f"adj_{prefix_name}")).tocoo()
    torch.save(
        {
            "features": getattr(result, f"feat_{prefix_name}").detach().cpu(),
            "labels": result.labels_train.cpu(),
            "edge_indices": torch.from_numpy(np.stack((graph.row, graph.col))),
            "edge_values": torch.from_numpy(graph.data),
            "shape": tuple(graph.shape),
        },
        payload,
    )
    metadata.write_text(json.dumps(expected, indent=2) + "\n")
    return result

"""Noise and PRBCD adversarial corruption, with split-safe source-keyed caches."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from graphslim.dataset.noise import cached_noise, graph_fingerprint
from graphslim.reduction.lightweight import as_csr
from graphslim.utils import seed_everything


def attack(data, args):
    if args.attack in (
        "random_adj",
        "random_feat",
        "edge_drop",
        "feature_mask",
        "gaussian",
        "label_noise",
    ):
        return cached_noise(data, args)
    if args.attack != "metattack":
        raise ValueError(f"Unknown attack {args.attack!r}")
    if not 0 <= args.ptb_r <= 1:
        raise ValueError("ptb_r must be in [0,1]")
    signature = {
        "kind": "metattack",
        "rate": args.ptb_r,
        "seed": args.seed,
        "setting": args.setting,
        "stage": "training",
    }
    if getattr(data, "noise_metadata", None) == signature:
        return data
    from graphslim.models import PRBCD
    from torch_geometric.data import Data

    seed_everything(args.seed)
    prefix = "full" if args.setting == "trans" else "train"
    features = torch.as_tensor(getattr(data, f"feat_{prefix}")).to(args.device)
    adjacency = as_csr(getattr(data, f"adj_{prefix}"))
    source = graph_fingerprint(data, args.setting)
    budget = {
        "epochs": getattr(args, "attack_epochs", 400),
        "pretrain_epochs": getattr(args, "attack_pretrain_epochs", 600),
        "search_space": getattr(args, "attack_search_space", 100000),
        "final_samples": getattr(args, "attack_final_samples", 20),
    }
    # Labels used by surrogate training/selection are also part of cache identity.
    from hashlib import sha256

    label_hash = sha256(
        torch.as_tensor(data.labels_train).cpu().numpy().tobytes()
        + torch.as_tensor(data.labels_val).cpu().numpy().tobytes()
    ).hexdigest()[:12]
    budget_hash = sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest()[:12]
    folder = Path(args.save_path) / "corrupt_graph/metattack"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_{args.ptb_r}_{args.seed}_{source[:16]}_{label_hash}_{budget_hash}"
    cache, metadata = folder / (name + ".npz"), folder / (name + ".json")
    expected = {
        **signature,
        "source_sha256": source,
        "budget": budget,
        "implementation": "PRBCD, not a literal MetaAttack",
    }
    if cache.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == expected:
        modified = sp.load_npz(cache)
        if modified.shape != adjacency.shape:
            raise ValueError("Adversarial cache shape mismatch")
    elif int(args.ptb_r * adjacency.nnz // 2) == 0:
        modified = adjacency
    else:
        n = len(features)
        train_mask, val_mask = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        labels = torch.zeros(n, dtype=torch.long)
        if args.setting == "trans":
            train_mask[data.idx_train] = True
            val_mask[data.idx_val] = True
            labels[data.idx_train] = torch.as_tensor(data.labels_train)
            labels[data.idx_val] = torch.as_tensor(data.labels_val)
            # Test nodes get placeholders; PRBCD replaces non-train labels by
            # surrogate predictions before optimization. Never read labels_test.
        else:
            heldout = max(1, min(n // 5, n - 1))
            train_mask[:-heldout], val_mask[-heldout:] = True, True
            labels[:] = torch.as_tensor(data.labels_train)
        surrogate_graph = Data(
            x=features,
            y=labels.to(args.device),
            edge_index=torch.tensor(
                np.stack(adjacency.nonzero()), dtype=torch.long, device=args.device
            ),
            train_mask=train_mask.to(args.device),
            val_mask=val_mask.to(args.device),
            test_mask=torch.zeros(n, dtype=torch.bool, device=args.device),
        )
        optimizer = PRBCD(
            surrogate_graph,
            device=args.device,
            epochs=budget["epochs"],
            fine_tune_epochs=min(100, budget["epochs"]),
            pretrain_epochs=budget["pretrain_epochs"],
            search_space_size=budget["search_space"],
            max_final_samples=budget["final_samples"],
            do_synchronize=False,
        )
        edge_index, _ = optimizer.attack(ptb_rate=args.ptb_r)
        edge_index = edge_index.cpu().numpy()
        modified = sp.csr_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float32), edge_index), shape=adjacency.shape
        )
        sp.save_npz(cache, modified)
        metadata.write_text(json.dumps(expected, indent=2) + "\n")
    result = deepcopy(data)
    setattr(result, f"adj_{prefix}", modified)
    if args.setting == "trans":
        result.edge_index = torch.tensor(np.stack(modified.nonzero()), dtype=torch.long)
        for split in ("train", "val", "test"):
            index = torch.as_tensor(getattr(result, f"idx_{split}")).cpu().numpy()
            setattr(result, f"adj_{split}", modified[index][:, index])
    for name in ("sparse_adj", "samplers", "class_dict", "class_dict2"):
        if hasattr(result, name):
            setattr(result, name, None)
    result.noise_metadata = signature
    return result

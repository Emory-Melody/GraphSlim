"""ClustGDD: propagated-attribute clustering and class-aware refinement.

Huang et al., KDD 2025, arXiv:2505.20807. Adapted algorithm components from
the MIT-licensed HKBU-LAGAS/ClustGDD release; see THIRD_PARTY_NOTICES.
"""

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from graphslim.acceleration import CUDAGraphStep

from graphslim.reduction.lightweight import (
    LightweightReducer,
    cluster,
    denoise,
    normalized_adjacency,
    sparse_tensor,
)


class LinearClassifier(nn.Module):
    """The author's no-ReLU, no-BN multi-layer linear classifier with dropout."""

    def __init__(self, input_size, hidden, classes, layers=2, dropout=0.6):
        super().__init__()
        dimensions = [input_size] + [hidden] * (layers - 1) + [classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dimensions, dimensions[1:]))
        self.dropout = dropout

    def forward(self, features):
        for pos, layer in enumerate(self.layers):
            features = layer(features)
            if pos < len(self.layers) - 1:
                features = F.dropout(features, self.dropout, self.training)
        return features


def consistency_loss(outputs, temperature=0.5):
    probabilities = torch.stack([output.softmax(1) for output in outputs])
    average = probabilities.mean(0)
    sharpened = average.clamp_min(1e-12).pow(1 / temperature)
    sharpened = (sharpened / sharpened.sum(1, keepdim=True)).detach()
    return temperature * (probabilities - sharpened).square().sum((1, 2)).mean()


def compress_graph(adjacency, assignments, size):
    """Column-normalized membership: C.T A C; no N x size dense membership."""
    counts = np.bincount(assignments, minlength=size)
    membership = sp.csr_matrix(
        (1 / counts[assignments], (np.arange(len(assignments)), assignments)),
        shape=(len(assignments), size),
    )
    compressed = (membership.T @ adjacency @ membership).tocsr()
    compressed.setdiag(0)
    compressed.eliminate_zeros()
    return compressed


def class_aware_views(adjacency, embeddings, ratio):
    """Class-wise attention x effective-resistance lower-bound top-edge sampling."""
    if not 0 <= ratio <= 1:
        raise ValueError("clustgdd_sampling_ratio must be in [0, 1]")
    adjacency = adjacency.coalesce()
    src, dst = adjacency.indices()
    similarity = F.cosine_similarity(embeddings[src], embeddings[dst], dim=1).clamp_min(0)
    values = adjacency.values() * similarity
    degrees = torch.zeros(len(embeddings), device=embeddings.device).scatter_add_(0, src, values)
    resistance = values / degrees[src].clamp_min(1e-12) + values / degrees[dst].clamp_min(1e-12)
    probabilities = embeddings.softmax(1)
    size = int(len(values) * ratio)
    views = []
    for label in range(embeddings.shape[1]):
        weights = probabilities[src, label] * probabilities[dst, label] * resistance
        chosen = torch.topk(weights, size).indices
        indices = adjacency.indices()[:, chosen].cpu().numpy()
        graph = sp.csr_matrix((values[chosen].cpu().numpy(), indices), shape=adjacency.shape)
        # The released directed top-k ties can break symmetry. Restore it for GNNs.
        views.append((graph + graph.T) * 0.5)
    return views


class ClustGDD(LightweightReducer):
    def reduce(self, data, verbose=True, save=True):
        args = self.args
        features, adjacency, train, labels = self.inputs(data)
        size = max(1, int(len(train) * args.reduction_rate))
        self.check_budget(size)
        normalized = normalized_adjacency(adjacency, self.device)
        steps = getattr(args, "clustgdd_prop_steps", 5)
        alpha = getattr(args, "clustgdd_alpha", 0.8)
        target = denoise(normalized, features, steps, alpha).detach()
        if self.setting == "trans":
            validation = target[torch.as_tensor(data.idx_val, device=self.device)]
        else:
            validation_features = torch.as_tensor(
                data.feat_val, dtype=torch.float32, device=self.device
            )
            validation = denoise(
                normalized_adjacency(data.adj_val, self.device), validation_features, steps, alpha
            )
        validation_labels = torch.as_tensor(data.labels_val, dtype=torch.long, device=self.device)

        def classifier():
            return LinearClassifier(
                features.shape[1],
                getattr(args, "clustgdd_hidden", 256),
                data.nclass,
                getattr(args, "clustgdd_layers", 2),
                getattr(args, "clustgdd_dropout", 0.6),
            ).to(self.device)

        model = classifier()
        learning_rate = getattr(args, "clustgdd_pre_lr", 0.01)
        pre_epochs = getattr(args, "clustgdd_pre_epochs", 80)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            # Author MLP(with_relu=False) forces effective pretrain decay to 0.
            weight_decay=getattr(args, "clustgdd_pre_wd", 0.0),
        )
        best_score, best_weights = -1, deepcopy(model.state_dict())
        training_features = target[train]
        pretraining_step = CUDAGraphStep(
            lambda: F.cross_entropy(model(training_features), labels),
            model.parameters(),
            args,
            "clustgdd.pretrain",
            pre_epochs,
            modules=(model,),
            inputs=(training_features, labels),
        )
        for epoch in range(pre_epochs):
            model.train()
            if epoch == pre_epochs // 2:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.1
            optimizer.zero_grad()
            pretraining_step()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                score = (
                    float((model(validation).argmax(1) == validation_labels).float().mean())
                    if len(validation_labels)
                    else -float(F.cross_entropy(model(target[train]), labels))
                )
                if score > best_score:
                    best_score, best_weights = score, deepcopy(model.state_dict())
        model.load_state_dict(best_weights)
        pretraining_validation = best_score
        model.eval()
        with torch.no_grad():
            embeddings = model(target)
        assignment = cluster(
            embeddings.cpu().numpy(), size, args.seed, minibatch=len(features) > 50000
        )
        assigned = torch.as_tensor(assignment, device=self.device)
        counts = torch.bincount(assigned, minlength=size).float()
        centers = (
            torch.zeros(size, features.shape[1], device=self.device).index_add_(0, assigned, target)
            / counts[:, None]
        )
        cluster_embedding = (
            torch.zeros(size, data.nclass, device=self.device).index_add_(0, assigned, embeddings)
            / counts[:, None]
        )
        synthetic_labels = cluster_embedding.argmax(1)
        adjacency_csr = sp.csr_matrix(
            (normalized.values().cpu().numpy(), normalized.indices().cpu().numpy()),
            shape=adjacency.shape,
        )
        reduced_csr = compress_graph(adjacency_csr, assignment, size)
        views = class_aware_views(
            normalized, embeddings, getattr(args, "clustgdd_sampling_ratio", 0.4)
        )
        compressed_views = [
            torch.as_tensor(
                compress_graph(view, assignment, size).toarray(),
                dtype=torch.float32,
                device=self.device,
            )
            for view in views
        ]
        offset = nn.Parameter(torch.zeros_like(centers))
        model = classifier()
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.parameters(),
                    "lr": getattr(args, "clustgdd_post_lr", 0.01),
                    "weight_decay": getattr(args, "clustgdd_post_wd", 5e-4),
                },
                {
                    "params": [offset],
                    "lr": getattr(args, "clustgdd_feature_lr", 1e-4),
                    "weight_decay": getattr(args, "clustgdd_post_wd", 5e-4),
                },
            ]
        )
        refinement = getattr(args, "clustgdd_refinement_scale", 0.01)
        best_score, best_offset = -1, offset.detach().clone()
        history = []

        def refinement_loss():
            refined = centers + refinement * offset
            predictions = [
                model(denoise(view, refined, getattr(args, "clustgdd_post_steps", 2), alpha))
                for view in compressed_views
            ]
            classification = F.cross_entropy(model(target[train]), labels)
            classification = classification + getattr(
                args, "clustgdd_prediction_weight", 1.0
            ) * sum(F.cross_entropy(output, synthetic_labels) for output in predictions)
            consistency = consistency_loss(predictions, getattr(args, "clustgdd_temperature", 0.5))
            return classification + getattr(args, "clustgdd_consistency_weight", 0.1) * consistency

        refinement_step = CUDAGraphStep(
            refinement_loss,
            list(model.parameters()) + [offset],
            args,
            "clustgdd.refinement",
            args.epochs,
            modules=(model,),
            inputs=(centers, target, synthetic_labels),
        )
        for epoch in range(args.epochs):
            model.train()
            if epoch == args.epochs // 2:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.1
            optimizer.zero_grad()
            loss = refinement_step()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ClustGDD refinement loss")
            optimizer.step()
            history.append(float(loss.detach()))
            with torch.no_grad():
                model.eval()
                score = (
                    float((model(validation).argmax(1) == validation_labels).float().mean())
                    if len(validation_labels)
                    else -history[-1]
                )
                if score > best_score:
                    best_score, best_offset = score, offset.detach().clone()
        reduced = torch.as_tensor(reduced_csr.toarray(), dtype=torch.float32, device=self.device)
        return self.finish(
            data,
            centers + refinement * best_offset,
            reduced,
            synthetic_labels,
            save=save,
            verbose=verbose,
            pretraining_steps=pre_epochs,
            refinement_steps=args.epochs,
            uses_validation_labels=True,
            pretraining_validation=pretraining_validation,
            refinement_validation=best_score,
            final_train_loss=history[-1],
        )

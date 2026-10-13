"""CGC: training-free class partition, calibration and feature reconstruction.

Independent implementation of Gao et al., WWW 2025 (arXiv:2405.13707).
Author reference: https://github.com/XYGaoG/CGC.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures

from graphslim.reduction.lightweight import (
    LightweightReducer,
    class_budget,
    cluster,
    normalized_adjacency,
)


def spectral_embedding(features, size, seed):
    """Low-rank degree-two polynomial affinity, without the N x N affinity."""
    centered = features - features.mean(0, keepdims=True)
    rank = min(size, centered.shape[0], centered.shape[1])
    if centered.shape[1] < 2 or not np.any(centered):
        return centered
    svd = TruncatedSVD(n_components=rank, random_state=seed)
    svd.fit(centered.T)
    basis = svd.components_.T
    polynomial = PolynomialFeatures(include_bias=True)
    mapped = polynomial.fit_transform(basis)
    powers = polynomial.powers_
    coefficients = np.ones(mapped.shape[1])
    coefficients[0] = 2**-0.5
    coefficients[(powers == 1).sum(1) == 2] = np.sqrt(2)
    coefficients[(powers == 1).sum(1) == 1] = np.sqrt(2 * 2**-0.5)
    mapped *= coefficients
    degree = mapped @ mapped.sum(0)
    mapped /= np.sqrt(np.maximum(degree, 1e-12))[:, None]
    rank = min(size + 1, mapped.shape[0], mapped.shape[1])
    svd = TruncatedSVD(n_components=rank, random_state=seed)
    svd.fit(mapped.T)
    return svd.components_.T[:, 1:]


def reconstruct_features(adjacency, embeddings, alpha=3.0, ridge=1e-8):
    """Released CGC solve: (A_conv A_conv.T + L) X = alpha A_conv.T H."""
    degree = adjacency.sum(1)
    inverse = (degree + 1e-10).rsqrt()
    laplacian = inverse[:, None] * (torch.diag(degree) - adjacency) * inverse[None, :]
    looped = adjacency + torch.eye(len(adjacency), device=adjacency.device)
    scale = looped.sum(1).rsqrt()
    normalized = scale[:, None] * looped * scale[None, :]
    convolution = torch.linalg.matrix_power(normalized, 3)
    system = convolution @ convolution.T + laplacian
    rhs = alpha * convolution.T @ embeddings
    # Solve in double precision; small jitter handles singular degenerate inputs.
    system = system.double() + ridge * torch.eye(
        len(system), device=system.device, dtype=torch.float64
    )
    return torch.linalg.solve(system, rhs.double()).to(embeddings.dtype)


class CGC(LightweightReducer):
    def reduce(self, data, verbose=True, save=True):
        args = self.args
        features, adjacency, train, labels = self.inputs(data)
        budget = class_budget(labels, args.reduction_rate)
        self.check_budget(sum(budget.values()))
        normalized = normalized_adjacency(adjacency, self.device)
        hop1 = normalized @ features
        hop2 = normalized @ hop1
        reference = ((features + hop1 + hop2) / 3)[train]
        targets = F.one_hot(labels, data.nclass).float()
        # CPU SVD lstsq matches the author's rank-deficient-safe gelss solver.
        weights = torch.linalg.lstsq(reference.cpu(), targets.cpu(), driver="gelss").solution.to(
            self.device
        )
        probabilities = (hop1[train] @ weights).softmax(1)
        augmentation = getattr(args, "cgc_aug_ratio", 0.5)
        rng = np.random.default_rng(args.seed)
        if augmentation > 0:
            f1 = f1_score(
                labels.cpu(),
                probabilities.argmax(1).cpu(),
                labels=list(range(data.nclass)),
                average=None,
                zero_division=0,
            )
            sampling = 1 - f1[labels.cpu().numpy()] + 1e-20
            sampling /= sampling.sum()
            sampled = rng.choice(len(labels), int(len(labels) * augmentation), p=sampling)
            sampled = torch.as_tensor(sampled, device=self.device)
            embeddings = torch.cat((hop2[train], hop1[train][sampled]))
            all_labels = torch.cat((labels, labels[sampled]))
        else:
            sampled = None
            embeddings, all_labels = hop2[train], labels

        def confidence(embedded, target):
            probability = (embedded @ weights).softmax(1)
            score = probability.gather(1, target[:, None]).flatten()
            return torch.where(probability.argmax(1) == target, score, score.min())

        scores = confidence(hop2[train], labels)
        if sampled is not None and len(sampled):
            # Match released calibration: set the minimum before selecting samples.
            scores = torch.cat((scores, confidence(hop1[train], labels)[sampled]))
        centers, synthetic_labels = [], []
        temperature = getattr(args, "cgc_tau", 1.0)
        if temperature <= 0:
            raise ValueError("cgc_tau must be positive")
        for label in sorted(budget):
            mask = all_labels == label
            values = embeddings[mask]
            embedding = values.detach().cpu().numpy()
            if getattr(args, "cgc_clustering", "spectral") == "spectral" and budget[label] > 1:
                embedding = spectral_embedding(embedding, budget[label], args.seed)
            assignment = cluster(embedding, budget[label], args.seed)
            assignment = torch.as_tensor(assignment, device=self.device)
            for index in range(budget[label]):
                members = assignment == index
                weight = (scores[mask][members] / temperature).softmax(0)
                centers.append((values[members] * weight[:, None]).sum(0))
                synthetic_labels.append(label)
        embeddings = torch.stack(centers)
        if getattr(args, "with_structure", 1):
            cosine = F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).T
            reduced_adjacency = (cosine > getattr(args, "cgc_adj_threshold", 0.8)).float()
            reduced_adjacency.fill_diagonal_(0)
            synthetic_features = reconstruct_features(
                reduced_adjacency, embeddings, getattr(args, "cgc_alpha", 7.0)
            )
        else:
            reduced_adjacency = torch.eye(len(embeddings), device=self.device)
            synthetic_features = embeddings
        return self.finish(
            data,
            synthetic_features,
            reduced_adjacency,
            torch.tensor(synthetic_labels, device=self.device),
            save=save,
            verbose=verbose,
            gnn_training_steps=0,
            clustering_backend="sklearn",
            variant="CGC" if getattr(args, "with_structure", 1) else "CGC-X",
        )

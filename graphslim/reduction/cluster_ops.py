"""Bounded distance, propagation and aggregation operators for clustering.

Clustering algorithms, initializers, restarts and stopping rules remain with
their callers. Sparse/segmented reductions can change floating-point sum order.
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor, matmul


def nearest_squared(points, centers, max_working_bytes=64 * 1024**2):
    """Exact subtract/square/sum distances, streamed with first-index tie breaks.

    Never construct N x K x D for the whole input. The budget covers the two
    largest subtraction/square temporaries; inputs and returned N-vectors are
    additional storage. No GEMM distance approximation or neighbor sampling.
    """
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("Expected matching two-dimensional feature matrices")
    if not len(centers) or not points.shape[1] or max_working_bytes < 1:
        raise ValueError("Nonempty centers/features and a positive memory budget required")
    if points.device != centers.device or points.dtype != centers.dtype:
        raise ValueError("Points and centers must share device and dtype")
    capacity = max(1, max_working_bytes // (2 * points.element_size() * points.shape[1]))
    center_block = min(len(centers), capacity)
    row_block = max(1, capacity // center_block)
    distances = points.new_full((len(points),), float("inf"))
    labels = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for start in range(0, len(points), row_block):
        stop = min(start + row_block, len(points))
        best, assigned = distances[start:stop], labels[start:stop]
        for first in range(0, len(centers), center_block):
            candidates = centers[first : first + center_block]
            squared = (points[start:stop, None] - candidates[None]).square().sum(2)
            value, index = squared.min(1)
            better = value < best
            assigned.copy_(torch.where(better, index + first, assigned))
            best.copy_(torch.minimum(best, value))
    return distances, labels


def fitted_labels(estimator, features, **fit_kwargs):
    """Reuse labels from sklearn's final assignment instead of predicting twice."""
    estimator.fit(features, **fit_kwargs)
    return np.asarray(estimator.labels_, dtype=np.int64)


def weighted_propagation(adjacency, features, weights):
    """Accumulate weighted sparse hops while retaining only the current hop."""
    result = torch.zeros_like(features)
    current = features
    for hop, weight in enumerate(weights):
        if hop:
            current = (
                matmul(adjacency, current)
                if isinstance(adjacency, SparseTensor)
                else adjacency @ current
            ).float()
        result.add_(weight * current)
    return result


def softmax_cluster_centers(features, scores, labels, size, temperature=1.0):
    """Segmented stable softmax and weighted sums, without K full-input masks."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = scores / temperature
    maximum = logits.new_full((size,), -float("inf"))
    maximum.scatter_reduce_(0, labels, logits, reduce="amax", include_self=True)
    weights = (logits - maximum[labels]).exp()
    denominator = weights.new_zeros(size).index_add_(0, labels, weights)
    weights = weights / denominator[labels]
    return features.new_zeros((size, features.shape[1])).index_add_(
        0, labels, features * weights[:, None]
    )


def sparse_membership(labels, size, weights=None, normalize=False):
    """K x N CSR membership with one entry per sample, replacing dense E/M."""
    labels = np.asarray(labels, dtype=np.int64)
    values = (
        np.ones(len(labels), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    matrix = sp.csr_matrix((values, (labels, np.arange(len(labels)))), shape=(size, len(labels)))
    if normalize:
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        if np.any(totals == 0):
            raise ValueError("Cannot normalize an empty cluster")
        matrix = sp.diags(1 / totals) @ matrix
    return matrix


def sparse_class_assignment(
    features,
    synthetic_labels,
    train_features,
    train_labels,
    train_indices,
    original_nodes,
    fuzzy=False,
):
    """Sparse GECC diagnostic assignment, indexed by original graph node IDs."""
    device = features.device
    train_labels = torch.as_tensor(train_labels, device=device)
    train_indices = torch.as_tensor(train_indices, device=device, dtype=torch.long)
    rows, columns, values = [], [], []
    for label in torch.unique(synthetic_labels, sorted=True).tolist():
        synthetic = (synthetic_labels == label).nonzero().flatten()
        original = (train_labels == label).nonzero().flatten()
        if not len(original):
            continue
        if fuzzy:
            candidates = train_features[original.to(train_features.device)].to(device)
            _, selected = nearest_squared(features[synthetic], candidates)
            rows.append(synthetic)
            columns.append(train_indices[original[selected]])
            values.append(features.new_ones(len(synthetic)))
        else:
            rows.append(synthetic.repeat_interleave(len(original)))
            columns.append(train_indices[original].repeat(len(synthetic)))
            values.append(features.new_full((len(synthetic) * len(original),), 1 / len(original)))
    indices = (
        torch.stack((torch.cat(rows), torch.cat(columns)))
        if rows
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    values = torch.cat(values) if values else features.new_empty(0)
    return torch.sparse_coo_tensor(indices, values, (len(features), original_nodes)).coalesce()

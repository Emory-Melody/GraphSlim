# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
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

    :param points: Torch feature matrix with shape ``(num_points, feature_dim)``.
    :param centers: Nonempty center matrix with the same feature width, dtype
        and device as ``points``.
    :param max_working_bytes: Positive working-memory target in bytes used to
        choose row and center blocks. At least one point-center comparison is
        always processed. Defaults to ``64 * 1024**2``.
    :returns: Tuple ``(squared_distances, center_indices)`` with one entry per
        point; ties select the first center index.
    :raises ValueError: If shapes, dtypes, devices or the memory budget are invalid.
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
    """Fit a clustering estimator and reuse its final assignments without a second prediction pass.

    :param estimator: Clustering estimator exposing ``fit`` and ``labels_``.
    :param features: Node feature matrix with one row per node.
    :param fit_kwargs: Keyword arguments passed to the estimator fit call.
    :returns: NumPy int64 array from the fitted estimator labels.
    """
    estimator.fit(features, **fit_kwargs)
    return np.asarray(estimator.labels_, dtype=np.int64)


def weighted_propagation(adjacency, features, weights):
    """Accumulate a weighted sum of feature propagation hops while storing only the current hop.

    :param adjacency: Graph adjacency; rows and columns refer to the same node ordering as the features.
    :param features: Node feature matrix with one row per node.
    :param weights: Propagation weights or sample weights, in the order consumed by this operation.
    :returns: Tensor ``sum(weights[h] * adjacency**h @ features)``.
    """
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
    """Compute clusterwise softmax-weighted feature centers with stable segmented reductions.

    :param features: Node feature matrix with one row per node.
    :param scores: Per-sample confidence logits used for within-cluster softmax weighting.
    :param labels: Integer cluster assignment for each input feature row.
    :param size: Number of output clusters.
    :param temperature: Positive softmax temperature applied to the sample scores. Defaults to ``1.0``.
    :returns: Tensor with one feature row per cluster.
    """
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
    """Build a cluster-by-sample sparse membership matrix with one stored value per sample.

    :param labels: Integer cluster assignment for each sample.
    :param size: Number of clusters.
    :param weights: Propagation weights or sample weights, in the order consumed by this operation. Defaults to ``None``.
    :param normalize: Whether to normalize each cluster row to unit total weight. Defaults to ``False``.
    :returns: SciPy CSR membership matrix; normalization rejects empty clusters.
    """
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
    """Construct sparse classwise assignments from synthetic nodes to original training node IDs.

    :param features: Node feature matrix with one row per node.
    :param synthetic_labels: Class labels of the synthetic feature rows.
    :param train_features: Training feature rows.
    :param train_labels: Class label for each training row.
    :param train_indices: Original graph node ID for each training row.
    :param original_nodes: Number of nodes in the original graph, defining the assignment matrix width.
    :param fuzzy: Whether to use distance-based soft assignments instead of uniform class memberships. Defaults to ``False``.
    :returns: Coalesced sparse COO tensor of shape ``(num_synthetic_nodes, original_nodes)``.
    """
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

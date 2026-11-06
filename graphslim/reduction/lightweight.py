# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Shared I/O for lightweight reducers; no validation/test labels enter reduction."""

from collections import Counter
import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from graphslim.reduction.cluster_ops import fitted_labels


def as_csr(adj):
    """Convert a dense, PyTorch sparse, torch_sparse, or SciPy adjacency to SciPy CSR.

    :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
    :returns: SciPy CSR matrix on the CPU.
    """
    if sp.issparse(adj):
        return adj.astype(np.float32).tocsr(copy=True)
    if isinstance(adj, torch.Tensor):
        if adj.layout != torch.strided:
            adj = adj.to_sparse_coo().coalesce().cpu()
            indices = adj.indices().numpy()
            return sp.csr_matrix((adj.values().numpy(), indices), shape=adj.shape)
        adj = adj.detach().cpu().numpy()
    elif hasattr(adj, "to_scipy"):
        return adj.to_scipy(layout="csr").astype(np.float32)
    return sp.csr_matrix(adj, dtype=np.float32)


def sparse_tensor(adj, device="cpu"):
    """Convert an adjacency to a coalesced PyTorch sparse COO tensor.

    :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
    :param device: Target PyTorch device, such as ``cpu`` or ``cuda:0``. Defaults to ``'cpu'``.
    :returns: Sparse float tensor on the requested device.
    """
    adj = as_csr(adj).tocoo()
    indices = torch.from_numpy(np.stack((adj.row, adj.col))).long()
    return (
        torch.sparse_coo_tensor(indices, torch.from_numpy(adj.data), adj.shape)
        .coalesce()
        .to(device)
    )


def normalized_adjacency(adj, device="cpu"):
    """Add one self-loop contribution per node and apply symmetric GCN normalization.

    :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
    :param device: Target PyTorch device, such as ``cpu`` or ``cuda:0``. Defaults to ``'cpu'``.
    :returns: Coalesced PyTorch sparse adjacency on the requested device.
    """
    adj = as_csr(adj)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(adj.sum(1)).ravel()
    inverse = np.zeros_like(degree)
    np.power(degree, -0.5, out=inverse, where=degree > 0)
    return sparse_tensor(sp.diags(inverse) @ adj @ sp.diags(inverse), device)


def propagate(adj, features, steps):
    """Apply repeated adjacency multiplication to node features.

    :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
    :param features: Node feature matrix with one row per node.
    :param steps: Number of adjacency propagation steps.
    :returns: Feature tensor after ``steps`` multiplications; zero steps returns the input.
    """
    for _ in range(steps):
        features = adj @ features
    return features


def denoise(adj, features, steps, alpha):
    """Evaluate the truncated PPR series ``(1-alpha) * sum((alpha*A)**t * X)`` for t from zero to steps-1.

    :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
    :param features: Node feature matrix with one row per node.
    :param steps: Number of adjacency propagation steps.
    :param alpha: Continuation weight in the truncated propagation series.
    :returns: Denoised feature tensor; requires ``steps >= 1`` and ``0 <= alpha < 1``.
    """
    if steps < 1 or not 0 <= alpha < 1:
        raise ValueError("Denoising needs steps >= 1 and 0 <= alpha < 1")
    current = features
    result = (1 - alpha) * current
    for _ in range(1, steps):
        current = alpha * (adj @ current)
        result = result + (1 - alpha) * current
    return result


def class_budget(labels, rate):
    """Allocate classwise synthetic node counts in rarest-class-first order.

    :param labels: Node labels in the same node ordering as the corresponding features.
    :param rate: Training-node retention fraction in the interval ``(0, 1]``.
    :returns: Dictionary of per-class counts, retaining at least one representative per observed class.
    """
    counts = Counter(labels.detach().cpu().tolist())
    if not counts or not 0 < rate <= 1:
        raise ValueError("Expected nonempty training labels and 0 < reduction_rate <= 1")
    target = max(len(counts), int(len(labels) * rate))
    allocation = {}
    remaining = target
    ordered = sorted(counts, key=lambda label: (counts[label], label))
    for pos, label in enumerate(ordered):
        reserved = len(ordered) - pos - 1
        size = remaining if reserved == 0 else max(int(counts[label] * rate), 1)
        size = min(counts[label], size, remaining - reserved)
        allocation[label] = size
        remaining -= size
    return allocation


def cluster(features, size, seed, *, minibatch=False):
    """Cluster feature rows with seeded k-means and repair empty or duplicate center assignments.

    :param features: Node feature matrix with one row per node.
    :param size: Size of the output or requested budget.
    :param seed: Random seed.
    :param minibatch: Whether to use MiniBatchKMeans instead of KMeans. Defaults to ``False``.
    :returns: Integer NumPy cluster labels with every requested cluster represented.
    """
    features = np.asarray(features, dtype=np.float32)
    if not 1 <= size <= len(features):
        raise ValueError("Cluster count must be between one and the number of samples")
    if size == 1:
        return np.zeros(len(features), dtype=np.int64)
    if size == len(features):
        return np.arange(size)
    estimator = (
        MiniBatchKMeans(n_clusters=size, batch_size=max(1000, size), random_state=seed, n_init=3)
        if minibatch
        else KMeans(n_clusters=size, random_state=seed, n_init=10, max_iter=300, tol=1e-4)
    )
    labels = fitted_labels(estimator, features)
    counts = np.bincount(labels, minlength=size)
    for empty in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        indices = np.flatnonzero(labels == donor)
        center = features[indices].mean(0)
        chosen = indices[np.argmax(((features[indices] - center) ** 2).sum(1))]
        labels[chosen] = empty
        counts[donor] -= 1
        counts[empty] += 1
    return labels.astype(np.int64)


class LightweightReducer:
    """The same constructor/reduce/save contract as other GraphSlim reducers."""

    def __init__(self, setting, data, args, **kwargs):
        """Initialize LightweightReducer and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        if setting not in ("trans", "ind"):
            raise ValueError("setting must be trans or ind")
        self.setting, self.args, self.device = setting, args, args.device
        self.started = time.perf_counter()

    def inputs(self, data):
        """Select the active graph and prepare device features, training indices, and labels.

        :param data: Graph data containing features, adjacency, labels, and split indices.
        :returns: Tuple ``(device_features, cpu_csr_adjacency, device_train_indices, device_training_labels)``.
        """
        if self.setting == "trans":
            features, adjacency = data.feat_full, data.adj_full
            indices = data.idx_train
        else:
            features, adjacency = data.feat_train, data.adj_train
            indices = torch.arange(len(features))
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(data.labels_train, dtype=torch.long, device=self.device)
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if labels.ndim != 1 or len(labels) != len(indices) or len(labels) == 0:
            raise ValueError("These methods require nonempty, single-class training labels")
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")
        if not 0 < self.args.reduction_rate <= 1:
            raise ValueError("reduction_rate must be in (0, 1]")
        return features, as_csr(adjacency), indices, labels

    def check_budget(self, size):
        """Reject a synthetic node budget that exceeds the configured safety limit.

        :param size: Size of the output or requested budget.
        """
        limit = getattr(self.args, "max_syn_nodes", 4096)
        if size > limit:
            raise ValueError(
                f"{size} synthetic nodes exceeds max_syn_nodes={limit}; lower the rate or explicitly raise the JSON limit (dense synthetic adjacency costs O(n^2))."
            )

    def finish(self, data, features, adjacency, labels, *, save, verbose, **metadata):
        """Validate synthetic tensors, attach reduction metadata, and optionally save them.

        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param features: Node feature matrix with one row per node.
        :param adjacency: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param labels: Node labels in the same node ordering as the corresponding features.
        :param save: Whether to write the reduced graph checkpoint to ``args.save_path``.
        :param verbose: Whether to print progress and diagnostic information.
        :param metadata: Method-specific metadata stored with the reduced graph.
        :returns: The input data with ``feat_syn``, ``adj_syn``, ``labels_syn``, and reduction metadata set.
        """
        data.feat_syn = features.detach().to(self.device, dtype=torch.float32)
        data.adj_syn = adjacency.detach().to(self.device, dtype=torch.float32)
        data.labels_syn = labels.detach().to(self.device)
        n = len(data.feat_syn)
        if n == 0 or data.adj_syn.shape != (n, n) or len(data.labels_syn) != n:
            raise ValueError("Invalid reduced graph dimensions")
        if not all(
            torch.isfinite(value).all() for value in (data.feat_syn, data.adj_syn, data.labels_syn)
        ):
            raise FloatingPointError("Reduction produced non-finite values")
        data.reduction_metadata = {
            "method": self.args.method,
            "setting": self.setting,
            "synthetic_nodes": n,
            "seconds": time.perf_counter() - self.started,
            "uses_test_labels": False,
            **metadata,
        }
        if save:
            from graphslim.dataset.utils import save_reduced

            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, self.args)
        if verbose:
            print(data.reduction_metadata)
        return data

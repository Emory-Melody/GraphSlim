"""Graph-property diagnostics, including tiny graphs and continuous targets."""

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import davies_bouldin_score
import torch

from graphslim.dataset.utils import get_syn_data
from graphslim.reduction.lightweight import as_csr


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def graph_statistics(adjacency, features, labels, structure=True):
    """Undefined quantities are None, never fabricated zeros or NaNs."""
    features, labels = _numpy(features), _numpy(labels)
    if labels.ndim == 2:
        labels = labels.argmax(1)
    adjacency = as_csr(adjacency)
    n = adjacency.shape[0]
    if len(features) != n or len(labels) != n:
        raise ValueError("Graph-property dimensions do not match")
    unique = len(np.unique(labels))
    valid_db = 1 < unique < n
    statistics = {
        "nodes": n,
        "davies_bouldin": float(davies_bouldin_score(features, labels)) if valid_db else None,
    }
    if not structure:
        return statistics
    # Self loops do not contribute to ordinary graph density or homophily.
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    graph = nx.from_scipy_sparse_array(adjacency)
    laplacian = nx.laplacian_matrix(graph).astype(np.float64)
    if laplacian.nnz == 0:
        radius, vector = 0.0, np.empty((n, 0))
    elif n <= 64:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        radius = float(values[-1]) if n else 0.0
        vector = vectors[:, -1:] if n else np.empty((0, 0))
    else:
        values, vector = eigsh(
            laplacian, k=1, which="LM", v0=np.random.default_rng(0).normal(size=n)
        )
        radius = float(values[0])
    projected = vector.T @ features
    trace = float(np.square(projected).sum() / features.shape[1]) if laplacian.nnz else None
    src, dst = adjacency.nonzero()
    looped = adjacency + sp.eye(n)
    degree = np.asarray(looped.sum(1)).ravel()
    inverse = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = sp.diags(inverse) @ looped @ sp.diags(inverse)
    aggregate = normalized @ normalized @ features
    statistics.update(
        edges=graph.number_of_edges(),
        density=float(nx.density(graph)),
        laplacian_trace=trace,
        spectral_radius=radius,
        clustering=float(nx.average_clustering(graph)),
        homophily=float(np.mean(labels[src] == labels[dst])) if len(src) else None,
        davies_bouldin_aggregated=float(davies_bouldin_score(aggregate, labels))
        if valid_db
        else None,
    )
    return statistics


class PropertyEvaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, data, reduced=True, model_type="GCN"):
        if reduced:
            self.feat, self.adj, self.label = get_syn_data(data, self.args, model_type=model_type)
        else:
            prefix = "train" if self.args.setting == "ind" else "full"
            self.feat = getattr(data, f"feat_{prefix}")
            self.adj = getattr(data, f"adj_{prefix}")
            self.label = getattr(data, f"labels_{prefix}")
        return (
            self.graph_property()
            if self.args.with_structure
            else self.graph_property_no_structure()
        )

    def _report(self, structure):
        adjacency = self.adj
        if isinstance(adjacency, torch.Tensor) and adjacency.ndim == 3:
            reports = [
                graph_statistics(a, self.feat, self.label[: len(a)], structure) for a in adjacency
            ]
        else:
            reports = graph_statistics(adjacency, self.feat, self.label, structure)
        self.args.logger.info("Graph properties: %s", reports)
        return reports

    def graph_property(self):
        return self._report(True)

    def graph_property_no_structure(self):
        return self._report(False)

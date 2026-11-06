"""Graph neural tangent kernel invariances and proximity-measure sanity checks."""

import numpy as np
import pytest
import scipy.sparse as sp

from graphslim.coarsening import utils
from graphslim.models.gntk import GNTK


@pytest.mark.parametrize("scale", ["uniform", "degree"])
@pytest.mark.parametrize("jumping", [False, True])
def test_graph_tangent_kernel_is_symmetric_and_node_permutation_invariant(scale, jumping):
    features = np.array([[1.0, 0.2], [0.2, 1], [1, 1], [0.6, 0.7]])
    adjacency = sp.csr_matrix([[1.0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]])
    permutation = np.array([2, 0, 3, 1])
    reordered = adjacency[permutation][:, permutation]
    kernel = GNTK(num_layers=3, num_mlp_layers=2, jk=jumping, scale=scale)
    first = kernel.diag(features, adjacency)
    second = kernel.diag(features[permutation], reordered)
    assert len(first) == (3 - 1) * 2
    self_value = kernel.gntk(features, features, first, first, adjacency, adjacency)[0]
    cross = kernel.gntk(features, features[permutation], first, second, adjacency, reordered)[0]
    reverse = kernel.gntk(features[permutation], features, second, first, reordered, adjacency)[0]
    assert np.isfinite(self_value) and self_value > 0
    assert cross == pytest.approx(reverse)
    assert cross == pytest.approx(self_value)


@pytest.mark.parametrize(
    "name", ["heavy_edge_degree", "rss", "rss_lanczos", "rss_cheby", "algebraic_GS"]
)
def test_proximity_measures_are_finite_for_connected_weighted_graph(name):
    from pygsp import graphs

    graph = graphs.Path(24)
    graph.compute_fourier_basis()
    np.random.seed(19)
    weights = utils.get_proximity_measure(graph, name, K=4)
    assert weights.shape == (graph.Ne,) and np.isfinite(weights).all()

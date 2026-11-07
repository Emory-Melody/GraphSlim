# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Numerical and independent-oracle checks for graph coarsening primitives."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
from pygsp import graphs

from graphslim.coarsening import utils
from graphslim.reduction import create_reducer


@pytest.mark.parametrize("maxcardinality", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_blossom_matching_attains_networkx_optimum(seed, maxcardinality):
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(5 + seed % 10, 0.25 + 0.1 * (seed % 5), seed=seed)
    for a, b in graph.edges:
        graph[a][b]["weight"] = int(rng.integers(-5, 25))
    edges = [(a, b, d["weight"]) for a, b, d in graph.edges(data=True)]
    mates = utils.maxWeightMatching(edges, maxcardinality=maxcardinality)
    pairs = [(i, j) for i, j in enumerate(mates) if j > i]
    assert len({i for pair in pairs for i in pair}) == 2 * len(pairs)
    assert all(graph.has_edge(a, b) and mates[b] == a for a, b in pairs)
    expected = nx.max_weight_matching(graph, maxcardinality=maxcardinality)
    assert sum(graph[a][b]["weight"] for a, b in pairs) == sum(
        graph[a][b]["weight"] for a, b in expected
    )
    if maxcardinality:
        assert len(pairs) == len(expected)


def test_empty_matching_and_unmatched_vertex():
    assert utils.maxWeightMatching([]) == []
    assert utils.maxWeightMatching([(1, 2, 7)]) == [-1, 2, 1]


def test_partition_projection_preserves_group_means_and_constant_vectors():
    graph = graphs.Path(6)
    c = utils.get_coarsening_matrix(graph, [[0, 1, 2], [3, 4]])
    np.testing.assert_allclose((c @ c.T).toarray(), np.eye(3))
    x = np.arange(6, dtype=float)
    np.testing.assert_allclose(utils.coarsen_vector(x, c), [1, 3.5, 5])
    np.testing.assert_allclose(utils.lift_vector(np.ones(3), c), np.ones(6))
    reduced = utils.coarsen_matrix(graph.W, c)
    lifted = utils.lift_matrix(reduced, c)
    assert reduced.shape == (3, 3) and lifted.shape == (6, 6)
    np.testing.assert_allclose(reduced.toarray(), reduced.T.toarray())
    np.testing.assert_allclose(lifted.toarray(), lifted.T.toarray())
    identity = sp.eye(6, format="csc")
    np.testing.assert_allclose(utils.coarsen_matrix(graph.W, identity).toarray(), graph.W.toarray())
    np.testing.assert_allclose(utils.lift_matrix(graph.W, identity).toarray(), graph.W.toarray())


@pytest.mark.parametrize("sparse", [False, True])
def test_zero_diagonal_preserves_offdiagonal_and_input(sparse):
    matrix = np.arange(16).reshape(4, 4).astype(float)
    value = sp.csr_matrix(matrix) if sparse else matrix.copy()
    actual = utils.zero_diag(value)
    if sparse:
        actual = actual.toarray()
    assert np.all(np.diag(actual) == 0)
    mask = ~np.eye(4, dtype=bool)
    np.testing.assert_array_equal(actual[mask], matrix[mask])
    np.testing.assert_array_equal(value.toarray() if sparse else value, matrix)


@pytest.mark.parametrize("order", ["ascend", "descend"])
def test_incidence_and_eigenvectors_reconstruct_laplacian(order):
    graph = graphs.Path(12)
    incidence = utils.get_S(graph)
    np.testing.assert_allclose(incidence @ incidence.T, graph.L.toarray())
    vectors, values = utils.eig(graph.L.toarray(), order=order)
    np.testing.assert_allclose(vectors @ np.diag(values) @ vectors.T, graph.L.toarray(), atol=1e-12)
    assert (
        np.all(np.diff(values) >= -1e-12) if order == "ascend" else np.all(np.diff(values) <= 1e-12)
    )


@pytest.mark.parametrize("strategy", ["greedy", "optimal"])
@pytest.mark.parametrize(
    "method",
    [
        "heavy_edge",
        "algebraic_jc",
        "affinity_gs",
        "variation_edges",
        "variation_neighborhoods",
        "variation_cliques",
        "kron",
    ],
)
def test_multilevel_coarseners_produce_valid_partitions(method, strategy):
    graph = graphs.Ring(32, k=2)
    graph.set_coordinates("ring2D")
    args = SimpleNamespace(device="cpu", reduction_rate=0.5, coarsen_strategy=strategy)
    np.random.seed(19)
    c, reduced, levels, hierarchy = create_reducer(method, "trans", None, args).coarsen(graph)
    assert c.shape == (reduced.N, graph.N)
    assert 0 < reduced.N < graph.N
    np.testing.assert_allclose((c @ c.T).toarray(), np.eye(reduced.N), atol=1e-12)
    assert (np.asarray((c != 0).sum(0)) == 1).all()
    np.testing.assert_allclose(reduced.W.toarray(), reduced.W.T.toarray(), atol=1e-12)
    assert np.isfinite(reduced.W.data).all()
    assert len(hierarchy) == len(levels) + 1
    np.testing.assert_allclose(reduced.coords, utils.coarsen_vector(graph.coords, c))


@pytest.mark.parametrize("mode", ["neighborhood", "cliques", "edges", "triangles"])
def test_variation_contractions_are_disjoint_and_within_budget(mode):
    graph = graphs.Ring(20, k=2)
    partition = utils.contract_variation_linear(graph, K=4, r=0.3, mode=mode)
    nodes = [int(i) for group in partition for i in group]
    assert len(nodes) == len(set(nodes))
    assert all(len(group) > 1 for group in partition)
    assert sum(len(group) - 1 for group in partition) <= 6


@pytest.mark.parametrize("method", ["GS", "JC", "Chebychev"])
def test_test_vector_smoothing_reduces_graph_dirichlet_energy(method):
    graph = graphs.Ring(20)
    np.random.seed(5)
    original = np.random.randn(graph.N, 4) / np.sqrt(graph.N)
    np.random.seed(5)
    smooth = utils.generate_test_vectors(graph, 4, method=method, iterations=5)
    assert smooth.shape == (20, 4) and np.isfinite(smooth).all()
    before = np.trace(original.T @ graph.L @ original)
    after = np.trace(smooth.T @ graph.L @ smooth)
    assert after <= before + 1e-10

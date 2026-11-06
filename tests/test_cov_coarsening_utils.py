"""Coverage-oriented checks of graphslim.coarsening.utils against independent oracles."""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
from pygsp import graphs

from graphslim.coarsening import utils

# Classic blossom stress cases (Van Rantwijk's mwmatching suite): nested
# S-blossoms, S-blossoms relabelled as T and expanded, and augmenting paths
# that run through expanded blossoms.
BLOSSOM_CASES = [
    [(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)],
    [(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7), (1, 6, 5), (4, 5, 6)],
    [(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 4), (1, 6, 3)],
    [(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 3), (1, 6, 4)],
    [(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 3), (3, 6, 4)],
    [(1, 2, 9), (1, 3, 9), (2, 3, 10), (2, 4, 8), (3, 5, 8), (4, 5, 10), (5, 6, 6)],
    [
        (1, 2, 10),
        (1, 7, 10),
        (2, 3, 12),
        (3, 4, 20),
        (3, 5, 20),
        (4, 5, 25),
        (5, 6, 10),
        (6, 7, 10),
        (7, 8, 8),
    ],
    [
        (1, 2, 8),
        (1, 3, 8),
        (2, 3, 10),
        (2, 4, 12),
        (3, 5, 12),
        (4, 5, 14),
        (4, 6, 12),
        (5, 7, 12),
        (6, 7, 14),
        (7, 8, 12),
    ],
    [
        (1, 2, 23),
        (1, 5, 22),
        (1, 6, 15),
        (2, 3, 25),
        (3, 4, 22),
        (4, 5, 25),
        (4, 8, 14),
        (5, 7, 13),
    ],
    [
        (1, 2, 19),
        (1, 3, 20),
        (1, 8, 8),
        (2, 3, 25),
        (2, 4, 18),
        (3, 5, 18),
        (4, 5, 13),
        (4, 7, 7),
        (5, 6, 7),
    ],
    [
        (1, 2, 45),
        (1, 5, 45),
        (2, 3, 50),
        (3, 4, 45),
        (4, 5, 50),
        (1, 6, 30),
        (3, 9, 35),
        (4, 8, 35),
        (5, 7, 26),
        (9, 10, 5),
    ],
    [
        (1, 2, 45),
        (1, 5, 45),
        (2, 3, 50),
        (3, 4, 45),
        (4, 5, 50),
        (1, 6, 30),
        (3, 9, 35),
        (4, 8, 26),
        (5, 7, 40),
        (9, 10, 5),
    ],
    [
        (1, 2, 45),
        (1, 5, 45),
        (2, 3, 50),
        (3, 4, 45),
        (4, 5, 50),
        (1, 6, 30),
        (3, 9, 35),
        (4, 8, 28),
        (5, 7, 26),
        (9, 10, 5),
    ],
    [
        (1, 2, 45),
        (1, 7, 45),
        (2, 3, 50),
        (3, 4, 45),
        (4, 5, 95),
        (4, 6, 94),
        (5, 6, 94),
        (6, 7, 50),
        (1, 8, 30),
        (3, 11, 35),
        (5, 9, 36),
        (7, 10, 26),
        (11, 12, 5),
    ],
    [
        (1, 2, 40),
        (1, 3, 40),
        (2, 3, 60),
        (2, 4, 55),
        (3, 5, 55),
        (4, 5, 50),
        (1, 8, 15),
        (5, 7, 30),
        (7, 6, 10),
        (8, 10, 10),
        (4, 9, 30),
    ],
    # Negative weights and a maximum-cardinality preference.
    [(1, 2, 2), (1, 3, -2), (2, 3, 1), (2, 4, -1), (3, 4, -6)],
]


def _oracle_weight(edges, maxcardinality):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    matching = nx.max_weight_matching(graph, maxcardinality=maxcardinality)
    return sum(graph[a][b]["weight"] for a, b in matching), len(matching)


def _matched_weight(edges, mates):
    weight = {frozenset((i, j)): w for i, j, w in edges}
    pairs = [(i, j) for i, j in enumerate(mates) if j > i]
    assert len({v for pair in pairs for v in pair}) == 2 * len(pairs)
    return sum(weight[frozenset(pair)] for pair in pairs), len(pairs)


@pytest.fixture
def checked_matching(monkeypatch):
    """Enable the module's internal dual-feasibility and optimality self-checks."""
    trace = []
    monkeypatch.setattr(utils, "CHECK_DELTA", True)
    monkeypatch.setattr(utils, "CHECK_OPTIMUM", True)
    monkeypatch.setattr(utils, "DEBUG", trace.append)
    return trace


@pytest.mark.parametrize("maxcardinality", [False, True])
@pytest.mark.parametrize("case", range(len(BLOSSOM_CASES)))
def test_blossom_stress_cases_reach_optimum_with_internal_checks(
    checked_matching, case, maxcardinality
):
    edges = BLOSSOM_CASES[case]
    mates = utils.maxWeightMatching(edges, maxcardinality=maxcardinality)
    assert all(m == -1 or mates[m] == v for v, m in enumerate(mates))
    weight, size = _matched_weight(edges, mates)
    expected_weight, expected_size = _oracle_weight(edges, maxcardinality)
    assert weight == expected_weight
    if maxcardinality:
        assert size == expected_size
    # The DEBUG hook receives a trace of stages; it must be non-empty strings.
    assert checked_matching and all(isinstance(line, str) for line in checked_matching)


@pytest.mark.parametrize("seed", range(25))
def test_random_integer_matchings_pass_internal_checks(checked_matching, seed):
    rng = np.random.default_rng(100 + seed)
    graph = nx.gnp_random_graph(14, 0.35, seed=200 + seed)
    edges = [(a, b, int(rng.integers(1, 30))) for a, b in graph.edges]
    if not edges:
        pytest.skip("empty random graph")
    maxcardinality = bool(seed % 2)
    mates = utils.maxWeightMatching(edges, maxcardinality=maxcardinality)
    assert _matched_weight(edges, mates)[0] == _oracle_weight(edges, maxcardinality)[0]


def test_float_weights_use_half_slack_branch(checked_matching):
    edges = [(0, 1, 1.5), (1, 2, 2.25), (2, 0, 2.0), (2, 3, 0.75), (3, 4, 1.25), (4, 5, 3.5)]
    mates = utils.maxWeightMatching(edges)
    assert _matched_weight(edges, mates)[0] == pytest.approx(_oracle_weight(edges, False)[0])


def test_projection_matrix_rows_are_orthonormal_and_lift_inverts_coarsen():
    graph = graphs.Ring(10)
    partition = [[0, 1, 2], [5, 6]]
    c = utils.get_coarsening_matrix(graph, partition)
    assert c.shape == (7, 10) and sp.issparse(c)
    np.testing.assert_allclose((c @ c.T).toarray(), np.eye(7), atol=1e-12)
    # Every original node belongs to exactly one coarse node.
    np.testing.assert_array_equal(np.asarray((c != 0).sum(0)).ravel(), np.ones(10))
    signal = np.arange(7, dtype=float)
    np.testing.assert_allclose(utils.coarsen_vector(utils.lift_vector(signal, c), c), signal)


def test_coarsening_quality_reports_reduction_and_spectral_errors():
    graph = graphs.Path(16)
    c = utils.get_coarsening_matrix(graph, [[2 * i, 2 * i + 1] for i in range(8)])
    coarse_laplacian = (c @ graph.L @ c.T).toarray()
    metrics = utils.coarsening_quality(graph, c, kmax=3)
    assert metrics["r"] == 0.5
    # Pairing a path into 8 supernodes leaves a path with 7 edges.
    assert metrics["m"] == 7
    assert metrics["error_eigenvalue"].shape == (3,) and metrics["error_eigenvalue"][0] == 0
    coarse_spectrum = np.linalg.eigvalsh(coarse_laplacian)[:3]
    original = np.linalg.eigvalsh(graph.L.toarray())[:3]
    np.testing.assert_allclose(
        metrics["error_eigenvalue"][1:],
        np.abs(original[1:] - coarse_spectrum[1:]) / original[1:],
        rtol=1e-4,
    )
    assert metrics["error_subspace"].shape == (3,) and metrics["error_sintheta"].shape == (3,)
    assert np.isfinite(metrics["error_subspace"]).all()
    assert (metrics["error_sintheta"] >= 0).all()


def test_coarsening_quality_accepts_precomputed_spectrum():
    graph = graphs.Path(12)
    c = utils.get_coarsening_matrix(graph, [[0, 1], [4, 5], [8, 9]])
    values, vectors = np.linalg.eigh(graph.L.toarray())
    metrics = utils.coarsening_quality(graph, c, kmax=3, Uk=vectors, lk=values.copy())
    assert metrics["r"] == pytest.approx(0.25)
    assert metrics["angle_matrix"].shape == (12, 3)
    assert np.isfinite(metrics["error_subspace"]).all()


def test_variation_edge_contraction_greedy_and_optimal_are_valid_matchings():
    graph = graphs.Ring(20, k=2)
    rng = np.random.default_rng(3)
    basis = rng.normal(size=(20, 4))
    budget = int(np.ceil(0.3 * 20))
    for algorithm in ["greedy", "optimal"]:
        pairs = utils.contract_variation_edges(graph, A=basis, K=4, r=0.3, algorithm=algorithm)
        pairs = np.asarray(pairs)
        assert pairs.ndim == 2 and pairs.shape[1] == 2 and 0 < len(pairs) <= budget
        assert len(np.unique(pairs)) == pairs.size
        assert all(graph.W[i, j] > 0 for i, j in pairs)


def test_variation_linear_computes_basis_when_missing():
    graph = graphs.Ring(16, k=2)
    partition = utils.contract_variation_linear(graph, K=4, r=0.25, mode="edges")
    nodes = [int(v) for group in partition for v in group]
    assert len(nodes) == len(set(nodes)) and len(partition) > 0
    assert sum(len(group) - 1 for group in partition) <= 4


@pytest.mark.parametrize(
    ("name", "sign"),
    [
        ("heavy_edge", 1),
        ("algebraic_JC", 1),
        ("affinity_GS", 1),
        ("algebraic_GS", 1),
        ("rss", -1),
        ("rss_lanczos", -1),
        ("rss_cheby", -1),
        ("min_expected_loss_lanczos", -1),
    ],
)
def test_proximity_measures_have_documented_sign(name, sign):
    graph = graphs.Ring(20, k=2)
    graph.compute_fourier_basis()
    np.random.seed(11)
    values = utils.get_proximity_measure(graph, name, K=4)
    assert values.shape == (graph.Ne,) and np.isfinite(values).all()
    assert (sign * values >= 0).all()


def test_heavy_edge_proximity_normalizes_by_max_incident_weight():
    weights = np.array([[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 4], [0, 0, 4, 0]], dtype=float)
    graph = graphs.Graph(weights)
    a, b, w = graph.get_edge_list()
    values = utils.get_proximity_measure(graph, "heavy_edge")
    wmax = weights.max(0) + 1e-5
    np.testing.assert_allclose(values, w / np.maximum(wmax[a], wmax[b]), rtol=1e-6)


def test_unknown_proximity_name_gives_zero_proximity():
    graph = graphs.Ring(8)
    np.testing.assert_array_equal(utils.get_proximity_measure(graph, "kron"), np.zeros(8))


def test_unknown_test_vector_method_returns_none():
    assert utils.generate_test_vectors(graphs.Ring(8), 2, method="unknown") is None


def test_optimal_matching_skips_self_loops_and_keeps_lightest_pairs():
    weights = graphs.Path(8).W.tolil()
    weights[3, 3] = 1.0
    graph = graphs.Graph(weights.tocsr())
    a, b, _ = graph.get_edge_list()
    assert (a == b).any()
    proximity = np.arange(graph.Ne, dtype=float)
    pairs = utils.matching_optimal(graph, proximity, r=0.1)
    assert pairs.shape == (1, 2) and pairs[0, 0] != pairs[0, 1]
    # The single retained pair is the lightest edge of the optimal matching.
    edge_index = {frozenset((i, j)): k for k, (i, j) in enumerate(zip(a, b))}
    assert proximity[edge_index[frozenset(pairs[0])]] == 0


def test_greedy_matching_prefers_heavy_edges_and_stops_at_target():
    graph = graphs.Path(6)
    weights = np.array([5.0, 1.0, 4.0, 1.0, 3.0])
    pairs = utils.matching_greedy(graph, weights, r=0.5)
    assert sorted(map(sorted, pairs.tolist())) == [[0, 1], [2, 3], [4, 5]]
    assert len(utils.matching_greedy(graph, weights, r=0.2)) == 2


def test_kron_coarsening_sets_missing_coordinates_and_halves_graph():
    graph = graphs.Graph(graphs.Path(16).W)
    assert not hasattr(graph, "coords")
    np.random.seed(2)
    coarse, original = utils.kron_coarsening(graph, r=0.5)
    assert original is graph and graph.coords.shape == (16, 2)
    assert coarse.N == 8 and len(np.unique(coarse.mr["idx"])) == 8


def test_kron_quality_flags_inconsistent_coarse_graph_as_failed():
    graph = graphs.Path(16)
    broken = graphs.Path(8)
    broken.mr = {"idx": np.arange(4)}
    metrics = utils.kron_quality(graph, broken, kmax=3)
    assert metrics["failed"] and "error_subspace" not in metrics


def test_multiresolution_symmetrizes_directed_kron_output(monkeypatch):
    original = utils.reduction.kron_reduction

    def directed(graph, ind):
        reduced = original(graph, ind)
        if isinstance(reduced, graphs.Graph):
            weights = sp.triu(reduced.W).tocsr()
            reduced = graphs.Graph(weights, coords=reduced.coords)
        return reduced

    monkeypatch.setattr(utils.reduction, "kron_reduction", directed)
    graph = graphs.Ring(16)
    graph.compute_fourier_basis()
    pyramid = utils.my_graph_multiresolution(graph, 1, sparsify=False)
    child = pyramid[1]
    assert not child.is_directed() and child.N == 8
    np.testing.assert_allclose(child.W.toarray(), child.W.T.toarray())


def test_zero_diag_and_incidence_on_weighted_graph():
    weights = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
    graph = graphs.Graph(weights)
    incidence = utils.get_S(graph)
    np.testing.assert_allclose(incidence @ incidence.T, graph.L.toarray())
    dense = utils.zero_diag(weights + np.eye(3))
    np.testing.assert_array_equal(dense, weights)


def test_sparsify_validates_epsilon_for_raw_laplacian_input():
    laplacian = graphs.Path(16).L
    with pytest.raises(ValueError, match="Epsilon"):
        utils.graph_sparsify(laplacian, 0.1)

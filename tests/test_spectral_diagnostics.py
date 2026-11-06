"""Independent spectral identities and exact matching oracles on small graphs."""

import networkx as nx
import numpy as np
import pytest
from pygsp import graphs

from graphslim.coarsening import utils


@pytest.mark.parametrize("kmax", [3, 8])
@pytest.mark.parametrize("provided_spectrum", [False, True])
def test_kron_quality_preserves_spectrum_and_matches_coarse_eigenvalues(kmax, provided_spectrum):
    graph = graphs.Path(24)
    coarse, _ = utils.kron_coarsening(graph, r=0.5)
    eigenvalues, eigenvectors = np.linalg.eigh(graph.L.toarray())
    original = eigenvalues.copy()
    cached_spectrum = graph.e.copy()
    kwargs = {"Uk": eigenvectors, "lk": eigenvalues} if provided_spectrum else {}
    quality = utils.kron_quality(graph, coarse, kmax=kmax, **kwargs)
    assert not quality["failed"]
    assert quality["r"] == 0.5 and quality["m"] == coarse.Ne
    assert quality["error_eigenvalue"].shape == (kmax,)
    assert np.isfinite(quality["error_subspace"]).all()
    expected = np.linalg.eigvalsh(coarse.L.toarray())[:kmax]
    np.testing.assert_allclose(
        quality["error_eigenvalue"][1:],
        np.abs(original[1:kmax] - expected[1:]) / original[1:kmax],
        atol=1e-5,
    )
    np.testing.assert_array_equal(eigenvalues, original)
    np.testing.assert_array_equal(graph.e, cached_spectrum)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("maxcardinality", [False, True])
def test_nested_odd_cycle_matchings_attain_networkx_optimum(seed, maxcardinality):
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(18, 0.2 + 0.05 * (seed % 5), seed=seed + 40)
    for a, b in graph.edges:
        graph[a][b]["weight"] = float(rng.uniform(-2, 10))
    edges = [(a, b, d["weight"]) for a, b, d in graph.edges(data=True)]
    mates = utils.maxWeightMatching(edges, maxcardinality=maxcardinality)
    pairs = [(i, j) for i, j in enumerate(mates) if i < j]
    expected = nx.max_weight_matching(graph, maxcardinality=maxcardinality)
    assert len({v for pair in pairs for v in pair}) == 2 * len(pairs)
    assert sum(graph[a][b]["weight"] for a, b in pairs) == pytest.approx(
        sum(graph[a][b]["weight"] for a, b in expected)
    )
    if maxcardinality:
        assert len(pairs) == len(expected)


@pytest.mark.parametrize("name", ["heavy_edge_degree", "min_expected_loss_lanczos"])
def test_proximity_measures_are_finite_and_do_not_change_graph(name):
    graph = graphs.Path(24)
    graph.compute_fourier_basis()
    weights = graph.W.toarray().copy()
    np.random.seed(7)
    values = utils.get_proximity_measure(graph, name, K=3)
    assert values.shape == (graph.Ne,) and np.isfinite(values).all()
    np.testing.assert_array_equal(graph.W.toarray(), weights)
    if name == "heavy_edge_degree":
        a, b, edge_weights = graph.get_edge_list()
        np.testing.assert_allclose(values, graph.dw[a] + graph.dw[b] + 2 * edge_weights)
    elif name.startswith(("rss", "min_expected")):
        assert (values <= 0).all()

"""Exercise actual NetworKit algorithms, checkpoint reloads, and downstream fits."""

import inspect
import json
from copy import deepcopy
from types import SimpleNamespace

import networkit as nk
import numpy as np
import pytest
import torch
from click.testing import CliRunner

from graphslim.config import cli, get_args
from graphslim.dataset import load_reduced
from graphslim.evaluation import Evaluator
from graphslim.reduction import create_reducer, get_method_spec
from test_lightweight_methods import quick_args, tiny_graph


# Independent upstream constructors, including defaults supplied by GraphSlim.
BACKENDS = [
    ("random_edge", "RandomEdgeSparsifier", {}),
    ("g_spar", "JaccardSimilaritySparsifier", {}),
    ("local_degree", "LocalDegreeSparsifier", {}),
    ("scan", "SCANSparsifier", {}),
    ("simmelian_parametric", "SimmelianSparsifierParametric", {"maxRank": 10}),
    ("simmelian_non_parametric", "SimmelianSparsifierNonParametric", {}),
    ("quadrilateral_simmelian", "QuadrilateralSimmelianSparsifier", {}),
    ("simmelian_multiscale", "SimmelianMultiscaleSparsifier", {}),
    ("degree_multiscale", "DegreeMultiscaleSparsifier", {"degsToAttrValue": max}),
    ("local_similarity", "LocalSimilaritySparsifier", {}),
    ("multiscale", "MultiscaleSparsifier", {}),
    ("random_node_edge", "RandomNodeEdgeSparsifier", {}),
    ("forest_fire", "ForestFireSparsifier", {"burnProbability": 0.5, "targetBurntRatio": 1.0}),
    ("triangle", "TriangleSparsifier", {}),
    ("algebraic_distance", "AlgebraicDistanceSparsifier", {}),
]


@pytest.fixture(autouse=True)
def single_thread():
    torch_threads, nk_threads = torch.get_num_threads(), nk.getMaxNumberOfThreads()
    torch.set_num_threads(1)
    nk.setNumberOfThreads(1)
    yield
    torch.set_num_threads(torch_threads)
    nk.setNumberOfThreads(nk_threads)


def edges(edge_index):
    return {tuple(sorted(edge)) for edge in edge_index.t().tolist()}


def test_every_concrete_upstream_sparsifier_is_reachable():
    upstream = {
        name
        for name, cls in inspect.getmembers(nk.sparsification, inspect.isclass)
        if issubclass(cls, nk.sparsification.Sparsifier)
        and name not in ("Sparsifier", "LocalSparsifier")
    }
    assert upstream == {backend for _, backend, _ in BACKENDS}
    for method, backend, _ in BACKENDS:
        cls = get_method_spec(method).load_class()
        assert cls.networkit_class == backend
        import graphslim.sparsification as api

        assert getattr(api, cls.__name__) is cls
    assert get_method_spec("Jaccard-Similarity") is get_method_spec("g_spar")


@pytest.mark.parametrize(
    "method,backend,options", [item for item in BACKENDS if item[0] != "triangle"]
)
def test_matches_native_size_parameterization(tmp_path, method, backend, options):
    nk.setSeed(4, False)
    graph = nk.generators.ErdosRenyiGenerator(36, 0.3).generate()
    graph.indexEdges()
    args = quick_args(tmp_path, method, "trans")
    args.seed = 17
    nk.setSeed(17, True)
    expected = getattr(nk.sparsification, backend)(**options).getSparsifiedGraphOfSize(graph, 0.5)
    actual, _ = create_reducer(method, "trans", None, args).edge_cutter(graph)
    assert edges(actual) == {tuple(sorted(uv)) for uv in expected.iterEdges()}


@pytest.mark.parametrize(
    "method", ["random_edge", "random_node_edge", "forest_fire", "algebraic_distance"]
)
def test_seed_controls_native_randomness(method):
    graph = nk.generators.ErdosRenyiGenerator(40, 0.3).generate()
    args = SimpleNamespace(reduction_rate=0.5, seed=19)
    first, _ = create_reducer(method, "trans", None, args).edge_cutter(graph)
    nk.setSeed(99, True)
    second, _ = create_reducer(method, "trans", None, args).edge_cutter(graph)
    assert edges(first) == edges(second)


@pytest.mark.parametrize("method,backend,options", BACKENDS)
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_roundtrip_and_actual_gcn_fit(tmp_path, method, backend, options, setting):
    graph = tiny_graph()
    original_features, original_labels = graph.feat_full.clone(), graph.labels_full.clone()
    args = quick_args(tmp_path, method, setting)
    nk.setSeed(7, False)
    reduced = create_reducer(method, setting, graph, args).reduce(graph)
    adjacency, features, labels = load_reduced(args, graph)
    original = graph.adj_full if setting == "trans" else graph.adj_train
    assert adjacency.shape == original.shape
    dense = adjacency.to_dense()
    torch.testing.assert_close(dense, dense.T)
    assert torch.isfinite(dense).all()
    assert not np.any((dense.numpy() != 0) & (original.toarray() == 0))
    assert dense.count_nonzero().item() // 2 == reduced.reduced_edges
    assert reduced.reduction_metadata["networkit_class"] == backend
    torch.testing.assert_close(adjacency, reduced.adj_syn)
    torch.testing.assert_close(
        features, graph.feat_full if setting == "trans" else graph.feat_train
    )
    # The shared loader returns training targets, including when the adjacency
    # spans the full transductive graph; Evaluator indexes its training nodes.
    torch.testing.assert_close(labels, graph.labels_train)
    torch.testing.assert_close(graph.feat_full, original_features)
    torch.testing.assert_close(graph.labels_full, original_labels)
    mean, std = Evaluator(args).evaluate(deepcopy(reduced), "GCN", verbose=False)
    assert np.isfinite(mean) and np.isfinite(std)


@pytest.mark.parametrize("method,backend,options", BACKENDS)
@pytest.mark.parametrize("kind", ["empty", "tree", "zero_rate", "full_rate"])
def test_degenerate_inputs_and_budget_endpoints(method, backend, options, kind):
    graph = nk.Graph(8)
    if kind != "empty":
        for node in range(6):
            graph.addEdge(node, node + 1)
    graph.indexEdges()
    rate = {"zero_rate": 0, "full_rate": 1}.get(kind, 0.5)
    args = SimpleNamespace(reduction_rate=rate)
    result, _ = create_reducer(method, "trans", None, args).edge_cutter(graph)
    assert result.shape[0] == 2
    assert edges(result) <= set(graph.iterEdges())
    if kind in ("empty", "zero_rate"):
        assert result.shape == (2, 0)
    if kind == "full_rate":
        assert edges(result) == set(graph.iterEdges())


def test_triangle_budget_scores_and_ties():
    graph = nk.Graph(7, weighted=True)
    # One triangle followed by a tail and an isolated node.
    for u, v in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]:
        graph.addEdge(u, v, 2.5)
    args = SimpleNamespace(reduction_rate=0.5)
    reducer = create_reducer("triangle", "trans", None, args)
    selected, weights = reducer.edge_cutter(graph)
    assert edges(selected) == {(0, 1), (0, 2), (1, 2)}
    assert torch.all(weights == 2.5)
    args.reduction_rate = 1 / 3
    selected, _ = reducer.edge_cutter(graph)
    assert edges(selected) == {(0, 1), (0, 2)}


def test_constructor_options_cli_config_and_local_composition(tmp_path):
    config = tmp_path / "options.json"
    config.write_text(json.dumps({"sparsifier_options": {"maxRank": 3}}))
    common = ["-M", "simmelian_parametric", "-G", "-1", "--save_path", str(tmp_path)]
    args = get_args([*common, "--config", str(config)])
    assert create_reducer(args.method, "trans", None, args).sparsifier.maxRank == 3
    args = get_args(
        [*common, "--config", str(config), "--sparsifier_options", '{"maxRank": 7, "local": true}']
    )
    reducer = create_reducer(args.method, "trans", None, args)
    assert isinstance(reducer.sparsifier, nk.sparsification.LocalSparsifier)
    assert reducer.sparsifier.sparsifier.maxRank == 7
    args.reduction_rate = 0.5
    graph = nk.generators.ErdosRenyiGenerator(12, 0.5).generate()
    actual, _ = reducer.edge_cutter(graph)
    assert edges(actual) <= {tuple(sorted(uv)) for uv in graph.iterEdges()}


@pytest.mark.parametrize(
    "method,options",
    [
        ("simmelian_parametric", {"maxRank": -1}),
        ("simmelian_parametric", {"unknown": 1}),
        ("forest_fire", {"burnProbability": 0}),
        ("forest_fire", {"targetBurntRatio": 2}),
        ("algebraic_distance", {"numberSystems": 0}),
        ("algebraic_distance", {"omega": float("nan")}),
        ("algebraic_distance", {"norm": 1}),
        ("random_node_edge", {"above": "false"}),
        ("degree_multiscale", {"degsToAttrValue": "unknown"}),
        ("triangle", {"local": "false"}),
        ("triangle", []),
    ],
)
def test_invalid_options_fail_before_native_code(method, options):
    with pytest.raises(ValueError):
        create_reducer(method, "trans", None, SimpleNamespace(sparsifier_options=options))


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_rates_fail_before_native_code(rate):
    reducer = create_reducer("forest_fire", "trans", None, SimpleNamespace(reduction_rate=rate))
    with pytest.raises(ValueError):
        reducer.edge_cutter(nk.Graph(3))


@pytest.mark.parametrize("value", ["{broken", "[]", "null"])
def test_cli_rejects_nonobject_options(value):
    result = CliRunner().invoke(cli, ["--sparsifier_options", value])
    assert result.exit_code == 2 and "--sparsifier_options" in result.output

"""Coreset selectors, their reduction bases, and NetworKit sparsifier edge cases."""

from collections import Counter
from types import SimpleNamespace

import networkit as nk
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_sparse import matmul

from graphslim.dataset import load_reduced
from graphslim.reduction import create_reducer
from graphslim.sparsification import CentD, CentP, Herding, HerdingAgg, KCenter, KCenterAgg
from graphslim.sparsification.kcenter_sample import KCenterSample
from graphslim.utils import normalize_adj_tensor, to_tensor
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    torch_threads, nk_threads = torch.get_num_threads(), nk.getMaxNumberOfThreads()
    torch.set_num_threads(1)
    nk.setNumberOfThreads(1)
    yield
    torch.set_num_threads(torch_threads)
    nk.setNumberOfThreads(nk_threads)


def class_counts(labels):
    return dict(Counter(torch.as_tensor(labels).long().tolist()))


def check_budget(reducer, selected, labels):
    assert len(selected) == len(set(selected.tolist()))
    assert class_counts(labels[selected]) == reducer.num_class_dict


# -------------------------------------------------------------- model-free


def star_graph():
    """Training nodes 0..15; even nodes have degree that grows with their index."""
    graph = tiny_graph()
    n = graph.num_nodes
    rows, cols = [], []
    for hub in range(16):
        for leaf in range(16, 16 + hub % 8 + 1):
            rows += [hub, leaf]
            cols += [leaf, hub]
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adjacency.data[:] = 1
    graph.adj_full = adjacency.astype(np.float32)
    graph.adj_train = graph.adj_full[:16, :16]
    return graph


def test_inductive_degree_centrality_uses_training_subgraph_degrees(tmp_path):
    graph = star_graph()
    dense = np.ones((16, 16)) - np.eye(16)
    dense[0, 2] = dense[2, 0] = 0
    graph.adj_train = sp.csr_matrix(dense)
    args = quick_args(tmp_path, "cent_d", "ind")
    reducer = CentD("ind", graph, args)
    selected = reducer.select()
    check_budget(reducer, selected, graph.labels_train)
    # Nodes 0 and 2 lost one training edge each, so neither is selected.
    assert not {0, 2} & set(selected.tolist())


def test_existing_synthetic_labels_supply_class_budgets(tmp_path):
    graph = tiny_graph()
    graph.labels_syn = torch.tensor([0, 1, 1])
    graph.num_class_dict = {0: 1, 1: 2}
    for setting in ("trans", "ind"):
        args = quick_args(tmp_path, "cent_d", setting)
        reducer = CentD(setting, graph, args)
        assert reducer.num_class_dict == {0: 1, 1: 2}
        assert np.array_equal(np.asarray(reducer.idx_train), np.arange(16))
        check_budget(reducer, reducer.select(), graph.labels_full)


def test_model_based_reduction_saves_loadable_checkpoint(tmp_path):
    graph = tiny_graph()
    args = quick_args(tmp_path, "kcenter", "trans")
    result = KCenter("trans", graph, args).reduce(graph)
    adjacency, features, labels = load_reduced(args)
    assert torch.equal(features, result.feat_syn)
    assert torch.equal(labels.long(), result.labels_syn.long())
    assert torch.equal(adjacency.to_dense(), result.adj_syn.to_dense())


def test_degree_centrality_picks_highest_degree_training_nodes_per_class(tmp_path):
    graph = star_graph()
    args = quick_args(tmp_path, "cent_d", "trans")
    reducer = CentD("trans", graph, args)
    selected = reducer.select()
    check_budget(reducer, selected, graph.labels_full)
    degree = np.asarray(graph.adj_full.sum(1)).ravel()
    for label in (0, 1):
        candidates = [i for i in range(16) if i % 2 == label]
        chosen = [i for i in selected if i % 2 == label]
        assert min(degree[chosen]) >= max(degree[sorted(set(candidates) - set(chosen))])
    result = reducer.reduce(graph, verbose=True, save=False)
    assert torch.equal(result.feat_syn, graph.feat_full[selected])
    assert torch.equal(result.labels_syn, graph.labels_full[selected])
    expected = torch.tensor(graph.adj_full[np.ix_(selected, selected)].toarray())
    assert torch.equal(result.adj_syn.to_dense(), expected)


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_pagerank_centrality_selects_class_budgets_from_training_nodes(tmp_path, setting):
    graph = star_graph()
    args = quick_args(tmp_path, "cent_p", setting)
    reducer = CentP(setting, graph, args)
    scores = reducer.pagerank_algorithm()
    assert scores.shape == ((16,) if setting == "ind" else (32,))
    assert np.isfinite(scores).all() and (scores > 0).all()
    selected = reducer.select()
    labels = graph.labels_train if setting == "ind" else graph.labels_full
    check_budget(reducer, selected, labels)
    assert set(selected.tolist()) <= set(range(16))
    result = reducer.reduce(graph, save=False)
    features = graph.feat_train if setting == "ind" else graph.feat_full
    assert torch.equal(result.feat_syn, features[selected])


def nonprefix_graph():
    graph = tiny_graph()
    graph.idx_train = torch.tensor([9, 4, 17, 2, 29, 6, 25, 12])
    ids = graph.idx_train.numpy()
    graph.feat_train = graph.feat_full[ids]
    graph.labels_train = graph.labels_full[ids]
    graph.adj_train = graph.adj_full[np.ix_(ids, ids)]
    return graph


def test_pagerank_matches_networkx_on_an_irregular_graph(tmp_path):
    import networkx as nx

    graph = star_graph()
    args = quick_args(tmp_path, "cent_p", "trans")
    reducer = CentP("trans", graph, args)
    adjacency = graph.adj_full.copy()
    # Regression: scores were uniform (1/n) because P r was used instead of P^T r.
    scores = reducer.pagerank_algorithm(max_iterations=1000, convergence_threshold=1e-12)
    reference = nx.pagerank(
        nx.from_scipy_sparse_array(adjacency), alpha=0.85, tol=1e-12, max_iter=1000
    )
    # networkx spreads isolated (dangling) mass uniformly, which only rescales the
    # fixed point, so the normalized vectors agree.
    theirs = np.array([reference[i] for i in range(len(scores))])
    np.testing.assert_allclose(scores / scores.sum(), theirs, rtol=1e-6)
    assert len(np.unique(np.round(scores, 12))) > 2
    selected = reducer.select()
    for label in (0, 1):
        chosen = [i for i in selected if i % 2 == label]
        rest = [i for i in range(16) if i % 2 == label and i not in chosen]
        assert scores[chosen].min() >= scores[rest].max()


def propagated(adjacency, features):
    operator = normalize_adj_tensor(to_tensor(adjacency), sparse=True)
    return matmul(matmul(operator, operator), features).float()


@pytest.mark.parametrize("cls", [KCenterAgg, HerdingAgg])
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_aggregated_coresets_select_from_two_hop_features_with_identity_graph(
    tmp_path, capsys, cls, setting
):
    # Regression: non-prefix training IDs; inductive aggregation must propagate
    # training-subgraph features rather than the first rows of feat_full.
    graph = nonprefix_graph()
    args = quick_args(tmp_path, "kcenter", setting)
    args.agg = True
    reducer = cls(setting, graph, args)
    adjacency = graph.adj_train if setting == "ind" else graph.adj_full
    features = graph.feat_train if setting == "ind" else graph.feat_full
    aggregated = propagated(adjacency, features)
    selected = reducer.select(aggregated)
    labels = graph.labels_train if setting == "ind" else graph.labels_full
    check_budget(reducer, selected, labels)
    result = reducer.reduce(graph, verbose=True, save=False)
    assert "selected nodes: 4" in capsys.readouterr().out
    torch.testing.assert_close(result.feat_syn, aggregated[selected])
    assert torch.equal(result.adj_syn, torch.eye(4))
    assert torch.equal(result.labels_syn, labels[selected])


def test_herding_and_kcenter_aggregation_follow_their_greedy_rules(tmp_path):
    graph = tiny_graph()
    args = quick_args(tmp_path, "kcenter", "trans")
    embeds = torch.tensor([[0.0], [10.0], [1.0], [11.0], [5.0], [12.0], [9.0], [13.0]])
    graph.idx_train = torch.arange(8)
    graph.labels_train = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    args.reduction_rate = 0.5
    herding = HerdingAgg("trans", graph, args).select(embeds)
    # Class 0 values 0, 1, 5, 9 have mean 3.75: herding picks 5, then the
    # point making the running mean closest to 3.75 (1 or 0 -> 1 is |3 - 3.75|).
    assert herding[:2].tolist() == [4, 2]
    centers = KCenterAgg("trans", graph, args).select(embeds)
    # K-center starts nearest the mean (5) and then adds the farthest point (0).
    assert centers[:2].tolist() == [4, 0]


# -------------------------------------------------------------- model-based


@pytest.mark.parametrize("cls", [KCenter, Herding, KCenterSample])
def test_model_based_inductive_coresets_use_local_training_ids(tmp_path, cls):
    graph = tiny_graph()
    args = quick_args(tmp_path, "kcenter", "ind")
    reducer = cls("ind", graph, args)
    result = reducer.reduce(graph, save=False)
    assert len(result.feat_syn) == 8
    rows = [int(torch.where((graph.feat_train == row).all(1))[0]) for row in result.feat_syn]
    assert torch.equal(result.labels_syn.long(), graph.labels_train[rows])
    assert class_counts(result.labels_syn) == reducer.num_class_dict
    expected = torch.tensor(graph.adj_train[np.ix_(rows, rows)].toarray()).float()
    assert torch.equal(result.adj_syn.to_dense(), expected)


def test_kcenter_sample_starts_nearest_the_mean_then_adds_farthest(tmp_path):
    graph = tiny_graph()
    graph.idx_train = torch.arange(8)
    graph.labels_train = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    args = quick_args(tmp_path, "kcenter", "trans")
    reducer = KCenterSample("trans", graph, args)
    embeds = torch.tensor([[0.0], [10.0], [1.0], [11.0], [5.0], [12.0], [9.0], [13.0]])
    selected = reducer.select(embeds)
    # Class 1 (10, 11, 12, 13) has a tie at the mean; argsort takes 11, then 13.
    assert selected.tolist() == [4, 0, 3, 7]


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_fixed_sfgc_indices_are_loaded_instead_of_training(tmp_path, monkeypatch, setting):
    import graphslim.sparsification.model_based_coreset_base as module

    graph = tiny_graph()
    args = quick_args(tmp_path, "sfgc", setting)
    args.dataset = "toy"
    # Regression: indices are read from the package folder, not the working directory.
    folder = tmp_path / "fixed_idx"
    folder.mkdir()
    monkeypatch.setattr(module, "FIXED_IDX_DIR", folder)
    fixed = np.array([3, 0, 5, 2])
    np.save(folder / f"idx_toy_{args.reduction_rate}_kcenter_15.npy", fixed)

    def no_training(*args, **kwargs):
        pytest.fail("Fixed SFGC indices must not train an embedding model")

    model_class = module.get_model_class("GCN")
    monkeypatch.setattr(model_class, "fit_with_val", no_training)
    result = KCenter(setting, graph, args).reduce(graph, save=False)
    features = graph.feat_train if setting == "ind" else graph.feat_full
    assert torch.equal(result.feat_syn, features[fixed])
    labels = graph.labels_train if setting == "ind" else graph.labels_full
    assert torch.equal(result.labels_syn.long(), labels[fixed])


def test_packaged_sfgc_indices_load_independently_of_working_directory(tmp_path, monkeypatch):
    from graphslim.sparsification.model_based_coreset_base import fixed_kcenter_indices

    monkeypatch.chdir(tmp_path)
    selected = fixed_kcenter_indices(SimpleNamespace(dataset="cora", reduction_rate=0.5))
    assert selected.ndim == 1 and len(selected) == len(set(selected.tolist())) > 0


# ------------------------------------------------------------ edge sparsifiers


def edges(edge_index):
    return {tuple(sorted(edge)) for edge in edge_index.t().tolist()}


def path_graph(n=8, weighted=False):
    graph = nk.Graph(n, weighted=weighted)
    for node in range(n - 1):
        graph.addEdge(node, node + 1, 2.0 if weighted else 1.0)
    graph.indexEdges()
    return graph


def test_sparsifier_options_accept_json_and_attribute_objects():
    args = SimpleNamespace(sparsifier_options='{"maxRank": 4}')
    assert create_reducer("simmelian_parametric", "trans", None, args).sparsifier.maxRank == 4
    args = SimpleNamespace(sparsifier_options=SimpleNamespace(maxRank=6))
    assert create_reducer("simmelian_parametric", "trans", None, args).sparsifier.maxRank == 6
    with pytest.raises(ValueError, match="JSON object"):
        create_reducer("triangle", "trans", None, SimpleNamespace(sparsifier_options="{bad"))


def test_directed_graphs_and_empty_budgets_are_handled():
    reducer = create_reducer("triangle", "trans", None, SimpleNamespace(reduction_rate=0.1))
    with pytest.raises(ValueError, match="undirected"):
        reducer.edge_cutter(nk.Graph(3, directed=True))
    selected, weights = reducer.edge_cutter(path_graph())
    assert selected.shape == (2, 0) and weights is None


def test_rank_degree_keeps_input_edge_weights(capsys):
    import random

    random.seed(0)
    args = SimpleNamespace(reduction_rate=0.5, verbose=False)
    graph = path_graph(12, weighted=True)  # rank degree samples 10 seed nodes
    selected, weights = create_reducer("rank_degree", "trans", None, args).edge_cutter(graph)
    assert edges(selected) <= set(graph.iterEdges())
    assert len(edges(selected)) >= int(11 * 0.5)
    assert torch.all(weights == 2.0)
    assert "targetNum: 5" in capsys.readouterr().out


def test_spanning_forest_keeps_one_tree_per_component():
    graph = nk.Graph(6)
    for u, v in [(0, 1), (1, 2), (2, 0), (3, 4)]:
        graph.addEdge(u, v)
    args = SimpleNamespace(verbose=False)
    selected, _ = create_reducer("spanning_forest", "trans", None, args).edge_cutter(graph)
    forest = edges(selected)
    assert len(forest) == 3 and (3, 4) in forest
    assert forest <= {(0, 1), (1, 2), (0, 2), (3, 4)}


def test_edge_sparsifier_without_saving_returns_cpu_adjacency(tmp_path):
    args = quick_args(tmp_path, "random_edge", "trans")
    args.seed = 1
    graph = tiny_graph()
    result = create_reducer("random_edge", "trans", graph, args).reduce(graph, save=False)
    assert result.adj_syn.shape == (32, 32)
    assert result.reduction_metadata["reduced_edges"] == result.reduced_edges
    assert result.reduced_edges <= result.reduction_metadata["original_edges"]
    assert not (tmp_path / "reduced_graph").exists()

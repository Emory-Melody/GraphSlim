"""Spectral, clustering, VNG and NOPE coarseners on small synthetic graphs."""

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from pygsp import graphs

from graphslim.coarsening import Average, Cluster, ClusterAgg, utils
from graphslim.coarsening.coarsening_base import Coarsen
from graphslim.coarsening.labels import label_tensor, synthetic_labels
from graphslim.coarsening.nope import NOPE, hierarchical_merge, interference_loss
from graphslim.coarsening.vng import VNG, layer_embeddings
from graphslim.dataset import load_reduced
from graphslim.reduction import create_reducer
from test_lightweight_methods import quick_args, tiny_graph

SPECTRAL = [
    "heavy_edge",
    "algebraic_jc",
    "affinity_gs",
    "kron",
    "variation_edges",
    "variation_neighborhoods",
    "variation_cliques",
]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seeded_arpack_start(monkeypatch):
    """Fix ARPACK's start vector so results do not depend on earlier eigsh calls.

    Without ``v0``, ARPACK draws its start vector from an internal seed that advances
    with every call in the process, so spectral coarsening varies with test order.
    """
    import scipy.sparse.linalg as linalg

    original = linalg.eigsh

    def eigsh(matrix, *args, **kwargs):
        if kwargs.get("v0") is None:
            kwargs["v0"] = np.random.default_rng(0).standard_normal(matrix.shape[0])
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(linalg, "eigsh", eigsh)


def coarsen(method, graph, rate=0.5, strategy="greedy"):
    args = SimpleNamespace(device="cpu", reduction_rate=rate, coarsen_strategy=strategy)
    reducer = create_reducer(method, "trans", None, args)
    np.random.seed(3)
    return reducer.coarsen(graph)


def assert_valid_hierarchy(graph, c, reduced, levels, hierarchy):
    assert c.shape == (reduced.N, graph.N)
    np.testing.assert_allclose((c @ c.T).toarray(), np.eye(reduced.N), atol=1e-10)
    assert (np.asarray((c != 0).sum(0)).ravel() == 1).all()
    assert len(hierarchy) == len(levels) + 1 and hierarchy[0] is graph
    sizes = [level.N for level in hierarchy]
    assert sizes == sorted(sizes, reverse=True)
    for level, matrix in zip(hierarchy, levels):
        assert matrix.shape[1] == level.N
    np.testing.assert_allclose(reduced.W.toarray(), reduced.W.T.toarray(), atol=1e-10)
    assert (reduced.W.diagonal() == 0).all()


@pytest.mark.parametrize("method", SPECTRAL)
def test_multilevel_coarsening_without_coordinates(method):
    graph = graphs.Graph(graphs.Ring(40, k=3).W)
    assert not hasattr(graph, "coords")
    c, reduced, levels, hierarchy = coarsen(method, graph, rate=0.25)
    assert_valid_hierarchy(graph, c, reduced, levels, hierarchy)
    assert not hasattr(reduced, "coords")
    assert 0 < reduced.N < graph.N and len(levels) >= 1


@pytest.mark.parametrize(
    ("method", "rate"),
    [("variation_edges", 0.1), ("variation_neighborhoods", 0.1), ("variation_cliques", 0.7)],
)
def test_variation_coarseners_descend_several_levels_to_reach_target(method, rate):
    graph = graphs.Ring(40)
    graph.set_coordinates("ring2D")
    c, reduced, levels, hierarchy = coarsen(method, graph, rate=rate)
    assert len(levels) >= 2
    assert reduced.N < hierarchy[1].N
    # Coarse coordinates are the projected original coordinates.
    np.testing.assert_allclose(reduced.coords, utils.coarsen_vector(graph.coords, c))


def test_reduction_rate_is_kept_fraction_for_edges_but_removed_fraction_for_cliques():
    # Documents the current (inconsistent) convention: variation_edges keeps
    # about rate * N nodes while variation_cliques removes about rate * N.
    graph = graphs.Path(40)
    _, edges, _, _ = coarsen("variation_edges", graph, rate=0.2)
    _, cliques, _, _ = coarsen("variation_cliques", graph, rate=0.2)
    assert edges.N <= 0.5 * graph.N and cliques.N >= 0.75 * graph.N


def star(leaves=11):
    weights = sp.lil_matrix((leaves + 1, leaves + 1))
    weights[0, 1:] = 1
    return graphs.Graph(weights.maximum(weights.T).tocsr())


@pytest.mark.parametrize("method", SPECTRAL)
def test_coarseners_stop_when_only_one_contraction_is_possible(method):
    # Every matching or disjoint contraction family of a star graph touches
    # the centre once, so the first level removes a single node and stops.
    graph = star()
    c, reduced, levels, hierarchy = coarsen(method, graph)
    assert reduced is graph and levels == [] and hierarchy == [graph]
    assert c.shape == (12, 12)


@pytest.mark.parametrize("method", ["algebraic_jc", "affinity_gs"])
def test_optimal_strategy_matches_valid_partition(method):
    graph = graphs.Ring(24, k=2)
    c, reduced, levels, hierarchy = coarsen(method, graph, rate=0.5, strategy="optimal")
    assert_valid_hierarchy(graph, c, reduced, levels, hierarchy)


def two_class_components(n_per=16, small=0):
    nodes = np.arange(n_per)
    ring = np.stack([nodes, np.roll(nodes, -1)])
    blocks = [ring, ring + n_per]
    total = 2 * n_per + small
    if small:
        tail = np.arange(2 * n_per, total)
        blocks.append(np.stack([tail[:-1], tail[1:]]))
    edges = np.concatenate(blocks, axis=1)
    adjacency = sp.csr_matrix((np.ones(edges.shape[1]), edges), shape=(total, total))
    adjacency = adjacency.maximum(adjacency.T)
    generator = torch.Generator().manual_seed(8)
    labels = torch.cat([torch.zeros(n_per), torch.ones(n_per), torch.zeros(small)]).long()
    return SimpleNamespace(
        x=torch.rand(total, 4, generator=generator),
        y=labels,
        adj_full=adjacency,
        adj_train=adjacency,
        idx_train=np.arange(total),
        train_mask=torch.ones(total, dtype=torch.bool),
        val_mask=torch.zeros(total, dtype=torch.bool),
    )


def coarsen_args(tmp_path, setting, method):
    return SimpleNamespace(
        device="cpu",
        setting=setting,
        reduction_rate=0.5,
        coarsen_strategy="greedy",
        save_path=str(tmp_path),
        method=method,
        attack=None,
        dataset="toy",
        seed=0,
        logger=logging.getLogger("coarsening-test"),
    )


def test_inductive_and_transductive_coincide_when_every_node_is_training(tmp_path):
    results = {}
    for setting in ["trans", "ind"]:
        data = two_class_components(small=3)
        args = coarsen_args(tmp_path, setting, "algebraic_jc")
        reducer = create_reducer("algebraic_jc", setting, data, args)
        np.random.seed(0)
        results[setting] = reducer.reduce(data, verbose=False, save=False)
    trans, ind = results["trans"], results["ind"]
    torch.testing.assert_close(trans.feat_syn, ind.feat_syn)
    torch.testing.assert_close(trans.labels_syn, ind.labels_syn)
    torch.testing.assert_close(trans.adj_syn.to_dense(), ind.adj_syn.to_dense())
    n = len(trans.feat_syn)
    # The 3-node component is below the size threshold and is dropped.
    assert 0 < n < 32 and trans.adj_syn.shape == (n, n)
    assert set(trans.labels_syn.tolist()) == {0, 1}
    dense = trans.adj_syn.to_dense()
    torch.testing.assert_close(dense, dense.T)


def test_transductive_coarsening_drops_mixed_and_unlabelled_supernodes(tmp_path):
    data = two_class_components()
    # Hide training labels on half of the second component.
    data.train_mask[16:24] = False
    args = coarsen_args(tmp_path, "trans", "variation_edges")
    np.random.seed(0)
    reduced = create_reducer("variation_edges", "trans", data, args).reduce(data, verbose=False)
    adjacency, features, labels = load_reduced(args)
    assert len(features) == len(labels) == adjacency.shape[0]
    assert set(labels.tolist()) <= {0, 1}
    torch.testing.assert_close(features, reduced.feat_syn)


def test_component_extraction_rejects_directed_and_skips_small_components():
    reducer = Coarsen("trans", None, SimpleNamespace(device="cpu"))
    directed = graphs.Graph(sp.triu(graphs.Ring(12).W).tocsr())
    with pytest.raises(NotImplementedError, match="Directed"):
        reducer.coarsening(directed)
    data = two_class_components(n_per=6, small=4)
    candidate, c_list, gc_list = reducer.coarsening(graphs.Graph(data.adj_full))
    assert [len(g.info["orig_idx"]) for g in candidate] == [6, 6, 4]
    assert c_list == [] and gc_list == []


def test_component_extraction_logs_and_returns_none_for_non_square_matrix(caplog):
    reducer = Coarsen("trans", None, SimpleNamespace(device="cpu"))
    fake = SimpleNamespace(A=sp.csr_matrix((3, 4)), logger=logging.getLogger("coarsen-test"))
    with caplog.at_level(logging.ERROR, logger="coarsen-test"):
        assert reducer.coarsening(fake) is None
    assert "Square matrix required" in caplog.text


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_graph_without_large_components_raises_clear_error(tmp_path, setting):
    # Regression: every component has <= 10 nodes, so nothing is coarsened;
    # previously trans failed with an object-array TypeError and ind with IndexError.
    data = two_class_components(n_per=6, small=4)
    args = coarsen_args(tmp_path, setting, "variation_edges")
    reducer = create_reducer("variation_edges", setting, data, args)
    with pytest.raises(ValueError, match="more than 10 nodes"):
        reducer.reduce(data, verbose=False, save=False)


@pytest.mark.parametrize("method", SPECTRAL)
def test_inductive_spectral_coarsening_uses_training_subgraph(tmp_path, method):
    # Regression: ind built the graph from adj_full but indexed training-only
    # features, raising IndexError for any node index >= number of training nodes.
    from test_legacy_reducer_integration import graph_fixture

    graph = graph_fixture()
    args = quick_args(tmp_path, method, "ind")
    np.random.seed(0)
    reduced = create_reducer(method, "ind", graph, args).reduce(graph, verbose=False)
    n_train = len(graph.idx_train)
    n = len(reduced.feat_syn)
    assert 0 < n < n_train and reduced.adj_syn.shape == (n, n)
    assert reduced.labels_syn.shape == (n,) and torch.isfinite(reduced.feat_syn).all()
    adjacency, features, labels = load_reduced(args)
    torch.testing.assert_close(features, reduced.feat_syn)


def test_heavy_edge_runs_through_reducer_pipeline(tmp_path):
    # Regression: Coarsen.coarsening calls coarsen(H) but HeavyEdge required ``method``.
    data = two_class_components()
    args = coarsen_args(tmp_path, "trans", "heavy_edge")
    reduced = create_reducer("heavy_edge", "trans", data, args).reduce(data, verbose=False)
    n = len(reduced.feat_syn)
    # Heavy-edge matching targets half of each class-pure 16-node ring.
    assert 16 <= n < 32 and reduced.adj_syn.shape == (n, n)
    assert set(reduced.labels_syn.tolist()) == {0, 1}
    dense = reduced.adj_syn.to_dense()
    torch.testing.assert_close(dense, dense.T)


def classwise_graph():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    features = torch.tensor(
        [[0.0, 0], [0, 1], [10, 10], [10, 11], [5, 5], [5, 6], [9, 9]], dtype=torch.float32
    )
    adjacency = sp.identity(7, format="csr")
    return SimpleNamespace(
        nclass=3,
        labels_train=labels,
        feat_train=features,
        feat_full=features,
        adj_full=adjacency,
        adj_train=adjacency,
        train_mask=torch.ones(7, dtype=torch.bool),
    )


@pytest.mark.parametrize("method", [Cluster, ClusterAgg])
def test_clustering_respects_preset_labels_and_skips_absent_classes(tmp_path, method):
    graph = classwise_graph()
    graph.labels_syn = torch.tensor([0, 0, 1])
    args = coarsen_args(tmp_path, "trans", "clustering")
    reduced = method("trans", graph, args).reduce(graph, verbose=False, save=True)
    assert reduced.labels_syn.tolist() == [0, 0, 1]
    assert reduced.feat_syn.shape == (3, 2) and torch.equal(reduced.adj_syn, torch.eye(3))
    adjacency, features, labels = load_reduced(args)
    torch.testing.assert_close(features, reduced.feat_syn)
    if method is Cluster:
        # Bisecting k-means splits class 0 into its two well separated pairs.
        rows = sorted(reduced.feat_syn[:2].tolist())
        np.testing.assert_allclose(rows, [[0.0, 0.5], [10.0, 10.5]])
        np.testing.assert_allclose(reduced.feat_syn[2], [5.0, 5.5])


@pytest.mark.parametrize("method", [Average, Cluster, ClusterAgg])
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_classwise_coarseners_derive_budgeted_labels(tmp_path, method, setting):
    graph = classwise_graph()
    args = coarsen_args(tmp_path, setting, "averaging")
    reduced = method(setting, graph, args).reduce(graph, verbose=False, save=True)
    # Budget is max(int(count * 0.5), 1) per class, rarest class first.
    assert reduced.labels_syn.tolist() == [2, 1, 0, 0]
    assert reduced.feat_syn.shape == (4, 2) and torch.equal(reduced.adj_syn, torch.eye(4))
    assert torch.isfinite(reduced.feat_syn).all()
    if method is Average:
        torch.testing.assert_close(reduced.feat_syn[0], graph.feat_train[6])
        torch.testing.assert_close(reduced.feat_syn[2], graph.feat_train[:4].mean(0))
    _, features, labels = load_reduced(args)
    torch.testing.assert_close(features, reduced.feat_syn)
    assert labels.tolist() == [2, 1, 0, 0]


def test_synthetic_labels_allocate_rarest_class_first_with_floor_one():
    labels = synthetic_labels(torch.tensor([2, 0, 0, 0, 0, 1, 1]), 0.5)
    assert labels.tolist() == [2, 1, 0, 0] and labels.dtype == np.int64
    tensor = label_tensor(labels, torch.zeros(1))
    assert tensor.dtype == torch.long and tensor.tolist() == [2, 1, 0, 0]


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_vng_reduce_with_default_sgc_teacher(tmp_path, setting):
    args = quick_args(tmp_path, "vng", setting)
    args.condense_model = "SGC"
    graph = tiny_graph()
    torch.manual_seed(0)
    reduced = VNG(setting, graph, args).reduce(graph, verbose=False)
    size = int(args.reduction_rate * 16)
    assert reduced.feat_syn.shape == (size, 8) and reduced.adj_syn.shape == (size, size)
    assert reduced.labels_syn.shape == (size,) and reduced.labels_syn.dtype == torch.long
    assert torch.isfinite(reduced.adj_syn).all() and torch.isfinite(reduced.feat_syn).all()
    adjacency, features, labels = load_reduced(args)
    torch.testing.assert_close(adjacency, reduced.adj_syn)


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_vng_reduce_with_multilayer_gcn_teacher(tmp_path, setting):
    # Regression: predict(..., output_layer_features=True) returns
    # (per_layer_features, log_probs); VNG iterated over the tuple itself.
    args = quick_args(tmp_path, "vng", setting)
    args.condense_model = "GCN"
    graph = tiny_graph()
    torch.manual_seed(0)
    reduced = VNG(setting, graph, args).reduce(graph, verbose=False)
    assert reduced.feat_syn.shape == (8, 8) and reduced.adj_syn.shape == (8, 8)
    assert torch.isfinite(reduced.adj_syn).all()


def test_layer_embeddings_keeps_single_tensor_model_output():
    features, log_probs = torch.ones(3, 2), torch.zeros(3, 2)
    assert [t is features for t in layer_embeddings((features, log_probs))] == [True, False]
    layers = [torch.ones(3, 4), torch.ones(3, 2)]
    assert layer_embeddings((layers, log_probs)) == layers


def test_vng_virtual_graph_solves_propagation_least_squares(capsys):
    rng = np.random.default_rng(4)
    n, features_dim = 12, 3
    features = torch.tensor(rng.normal(size=(n, features_dim)), dtype=torch.float32)
    embeds = [torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float32) for _ in range(2)]
    adjacency = sp.random(n, n, density=0.3, random_state=4, format="csr")
    adjacency = adjacency + adjacency.T
    labels = np.eye(2)[np.arange(n) % 2]
    agent = VNG("trans", None, SimpleNamespace(device="cpu", reduction_rate=0.5))
    edge, coarse_features, coarse_labels = agent.vng(
        features, embeds, adjacency, labels, verbose=True
    )
    assert "A_vr" in capsys.readouterr().out
    assert edge.shape == (6, 6) and coarse_features.shape == (6, features_dim)
    assert coarse_labels.shape == (6,) and set(coarse_labels.tolist()) <= {0, 1}
    # A_vr is the least-squares solution of A_vr @ P = Q, with P = E X_head.
    head = torch.cat(embeds, 1).numpy()
    weights = np.asarray(adjacency.sum(0)).ravel()
    weights[weights == 0] = 1
    from sklearn.cluster import KMeans

    from graphslim.reduction.cluster_ops import fitted_labels, sparse_membership

    assignment = fitted_labels(
        KMeans(n_clusters=6, random_state=2024, n_init=1), head, sample_weight=weights
    )
    membership = sparse_membership(assignment, 6, weights, normalize=True)
    p = membership @ head
    q = membership @ adjacency @ head
    np.testing.assert_allclose(edge.numpy() @ p, q, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(coarse_features.numpy(), membership @ features.numpy(), atol=1e-5)


def test_interference_losses_match_closed_forms():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    sizes = np.array([1.0, 3.0, 1.0, 1.0])
    neighbors = {0: {1, 2}, 1: {0, 2, 3}, 2: {0, 1}, 3: {1}}
    weight = 1 * 3 / (4 + 1e-8)
    relaxed = interference_loss(0, 1, features, sizes, neighbors, relaxed=True)
    assert relaxed == pytest.approx(weight * 2 * 2)
    exact = interference_loss(0, 1, features, sizes, neighbors)
    sum_u = 0 + 1  # neighbors 1 and 2 of node 0: (0)^2 + (1)^2
    sum_v = 0 + 1 + 0  # neighbors 0, 2, 3 of node 1: 0, 1, 0
    difference = 0.0  # common neighbor 2: <(1,1), (1,-1)> = 0
    assert exact == pytest.approx(weight * (sum_u + sum_v + difference))
    cached = interference_loss(0, 1, features, sizes, neighbors, sum_sq=np.array([5.0, 7.0]))
    assert cached == pytest.approx(weight * 12)


@pytest.mark.parametrize("relaxed", [False, True])
def test_hierarchical_merge_respects_components_and_target(relaxed):
    blocks = [np.ones((4, 4)) - np.eye(4), np.ones((3, 3)) - np.eye(3), np.zeros((1, 1))]
    adjacency = sp.block_diag(blocks, format="csr")
    features = np.arange(16, dtype=np.float32).reshape(8, 2)
    mapping, centers, tree = hierarchical_merge(features, adjacency, 2, relaxed)
    # Three components cannot be merged below three clusters.
    assert len(centers) == 3 and sorted(np.bincount(mapping).tolist()) == [1, 3, 4]
    assert len(set(mapping[:4])) == 1 and len(set(mapping[4:7])) == 1
    for cluster in range(3):
        np.testing.assert_allclose(centers[cluster], features[mapping == cluster].mean(0))
    assert len(tree) == 5
    identity, _, _ = hierarchical_merge(features, adjacency, 100, relaxed)
    np.testing.assert_array_equal(np.sort(identity), np.arange(8))


def test_hierarchical_merge_prunes_stale_heap_entries():
    n = 60
    rows = np.repeat(np.arange(n), n)
    cols = np.tile(np.arange(n), n)
    adjacency = sp.csr_matrix((np.ones(n * n), (rows, cols)), shape=(n, n))
    features = np.random.default_rng(1).normal(size=(n, 2)).astype(np.float32)
    mapping, centers, _ = hierarchical_merge(features, adjacency, 1, relaxed=True)
    assert len(centers) == 1 and (mapping == 0).all()
    np.testing.assert_allclose(centers[0], features.mean(0), rtol=1e-4, atol=1e-5)


def nope_args(tmp_path, setting, **overrides):
    args = quick_args(tmp_path, "cgc", setting)
    args.method = "nope"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("relaxed", [False, True])
def test_nope_reduces_to_labelled_supernodes(tmp_path, setting, relaxed, capsys):
    graph = tiny_graph()
    args = nope_args(tmp_path, setting, nope_relaxed=relaxed)
    reduced = NOPE(setting, graph, args).reduce(graph, verbose=True, save=True)
    meta = reduced.reduction_metadata
    assert meta["variant"] == ("NOPE*" if relaxed else "NOPE")
    assert meta["requested_nodes"] == 8 and meta["budget_reached"]
    n = len(reduced.feat_syn)
    assert n == meta["coarse_nodes"] - meta["unlabeled_supernodes_removed"]
    assert reduced.adj_syn.shape == (n, n) and set(reduced.adj_syn.unique().tolist()) <= {0, 1}
    torch.testing.assert_close(reduced.adj_syn, reduced.adj_syn.T)
    assert (reduced.adj_syn.diagonal() == 0).all()
    total = graph.num_nodes if setting == "trans" else 16
    assert reduced.nope_mapping.shape == (total,)
    assert "NOPE" in capsys.readouterr().out
    adjacency, features, labels = load_reduced(args)
    torch.testing.assert_close(features, reduced.feat_syn)


def test_nope_graph_budget_warns_when_components_block_target(tmp_path):
    graph = tiny_graph()
    adjacency = sp.lil_matrix((32, 32))
    for i in range(0, 32, 2):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    graph.adj_full = adjacency.tocsr()
    args = nope_args(tmp_path, "trans", nope_budget_base="graph", reduction_rate=0.25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reduced = NOPE("trans", graph, args).reduce(graph, verbose=False, save=False)
    assert any("disconnected components" in str(w.message) for w in caught)
    assert reduced.reduction_metadata["coarse_nodes"] == 16
    assert not reduced.reduction_metadata["budget_reached"]


def test_nope_rejects_unknown_budget_base(tmp_path):
    graph = tiny_graph()
    args = nope_args(tmp_path, "trans", nope_budget_base="nodes")
    with pytest.raises(ValueError, match="training or graph"):
        NOPE("trans", graph, args).reduce(graph, verbose=False, save=False)

"""Graph format conversion with duck-typed DGL graphs and on-disk edge lists."""

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from graphslim.dataset import convertor


class Edges:
    def __init__(self, by_type, data):
        self.by_type = by_type
        self.data_by_type = data

    def __call__(self, form="uv", etype=None):
        if etype is None:
            etype = next(iter(self.by_type))
        return self.by_type[etype]

    def __getitem__(self, etype):
        return type("EdgeView", (), {"data": self.data_by_type.get(etype, {})})()


class FakeDGL:
    def __init__(self, ntypes, counts, node_data, edges, edge_data):
        self.ntypes = ntypes
        self.counts = counts
        self.node_data = node_data
        self.canonical_etypes = list(edges)
        self.edges = Edges(edges, edge_data)
        self.is_homogeneous = len(ntypes) == 1 and len(edges) == 1
        self.ndata = node_data[ntypes[0]]
        self.edata = edge_data.get(self.canonical_etypes[0], {})
        self.nodes = {t: type("NodeView", (), {"data": node_data[t]})() for t in ntypes}

    def num_nodes(self, ntype=None):
        return self.counts[ntype or self.ntypes[0]]


def single_type(feature_key="feat", n_edge_types=1):
    features = torch.arange(8.0).reshape(4, 2)
    node_data = {"user": {feature_key: features, "label": torch.tensor([0, 1, 2, 1])}}
    etypes = [("user", f"rel{i}", "user") for i in range(n_edge_types)]
    edges = {
        etype: (torch.tensor([0, 1 + i]), torch.tensor([1 + i, 3]))
        for i, etype in enumerate(etypes)
    }
    edge_data = {etypes[0]: {"weight": torch.tensor([0.5, 2.0])}}
    return FakeDGL(["user"], {"user": 4}, node_data, edges, edge_data)


@pytest.mark.parametrize("feature_key", ["feature", "feat", "x"])
def test_homogeneous_dgl_graph_keeps_features_labels_and_weights(feature_key):
    graph = single_type(feature_key)
    data = convertor.from_dgl(graph, "toy")
    assert data.edge_index.tolist() == [[0, 1], [1, 3]]
    assert data.num_nodes == 4
    torch.testing.assert_close(data.x, torch.arange(8.0).reshape(4, 2))
    assert data.y.tolist() == [0, 1, 2, 1]
    torch.testing.assert_close(data.edge_weight, torch.tensor([0.5, 2.0]))


@pytest.mark.parametrize("feature_key", ["feature", "feat"])
def test_single_type_multi_relation_graph_flattens_with_self_loops(feature_key):
    graph = single_type(feature_key, n_edge_types=2)
    hetero = convertor.from_dgl(graph, "fraud")
    assert isinstance(hetero, HeteroData) and hetero.name == "fraud"
    assert hetero["user"].num_nodes == 4
    torch.testing.assert_close(hetero[graph.canonical_etypes[0]].weight, torch.tensor([0.5, 2.0]))
    flat = convertor.from_dgl(graph, "fraud", hetero=False)
    pairs = set(map(tuple, flat.edge_index.T.tolist()))
    assert {(0, 1), (1, 3), (0, 2), (2, 3)} <= pairs
    assert {(i, i) for i in range(4)} <= pairs
    assert flat.num_nodes == 4 and flat.num_classes == 3
    assert flat.y.tolist() == [0, 1, 2, 1]


def test_multiple_node_types_cannot_be_flattened_without_mapping():
    node_data = {"a": {"feat": torch.ones(2, 1)}, "b": {"feat": torch.ones(3, 1)}}
    edges = {("a", "to", "b"): (torch.tensor([0]), torch.tensor([2]))}
    graph = FakeDGL(["a", "b"], {"a": 2, "b": 3}, node_data, edges, {})
    hetero = convertor.from_dgl(graph, "bipartite")
    assert hetero["b"].num_nodes == 3
    assert hetero[("a", "to", "b")].edge_index.tolist() == [[0], [2]]
    with pytest.raises(ValueError, match="multiple node types"):
        convertor.from_dgl(graph, "bipartite", hetero=False)


def test_pygsp_graph_is_symmetric_dense_adjacency():
    graph = convertor.pyg2gsp(torch.tensor([[0, 1], [1, 2]]))
    assert graph.N == 3
    np.testing.assert_array_equal(graph.W.toarray(), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def test_batched_dense_tensor_uses_any_nonzero_channel():
    stacked = torch.zeros(3, 3, 2)
    stacked[0, 1, 1] = 4.0
    stacked[2, 0, 0] = -1.0
    sparse = convertor.dense2sparsetensor(stacked)
    assert sparse.sparse_sizes() == (3, 3)
    row, col, value = sparse.coo()
    assert list(zip(row.tolist(), col.tolist())) == [(0, 1), (2, 0)]
    torch.testing.assert_close(value, torch.tensor([[0.0, 4.0], [-1.0, 0.0]]))


def test_edge_list_file_is_read_as_undirected_indexed_graph(tmp_path):
    folder = tmp_path / "toy" / "raw"
    folder.mkdir(parents=True)
    (folder / "uduw.el").write_text("0 1\n1 2\n2 3\n")
    graph = convertor.loadSparseGraph("toy", str(tmp_path))
    assert graph.numberOfNodes() == 4 and graph.numberOfEdges() == 3
    assert not graph.isDirected() and graph.hasEdgeIds()
    assert graph.hasEdge(1, 0) and not graph.hasEdge(0, 3)


def test_homogeneous_graph_without_features_labels_or_weights_keeps_raw_attributes():
    edges = {("n", "e", "n"): (torch.tensor([0, 2]), torch.tensor([1, 0]))}
    node_data = {"n": {"degree": torch.tensor([1, 1, 1])}}
    edge_data = {("n", "e", "n"): {"kind": torch.tensor([3, 4])}}
    graph = FakeDGL(["n"], {"n": 3}, node_data, edges, edge_data)
    data = convertor.from_dgl(graph, "plain")
    assert data.num_nodes == 3 and data.x is None and data.y is None
    assert data.degree.tolist() == [1, 1, 1] and data.kind.tolist() == [3, 4]
    assert "edge_weight" not in data

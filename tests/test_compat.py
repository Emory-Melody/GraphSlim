from types import SimpleNamespace

import torch
import pytest

from graphslim.compat import to_pyg_data, to_dgl_graph


def test_to_pyg_data_accepts_existing_pyg_graph():
    from torch_geometric.data import Data

    data = Data(edge_index=torch.tensor([[0, 1], [1, 2]]), x=torch.ones(3, 2), y=torch.arange(3))

    assert to_pyg_data(data) is data


def test_to_pyg_data_accepts_reduced_graph_object():
    graph = SimpleNamespace(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        feat_syn=torch.ones(3, 2),
        labels_syn=torch.arange(3),
    )

    data = to_pyg_data(graph)

    assert data.edge_index.shape == (2, 2)
    assert data.x.shape == (3, 2)
    assert data.y.tolist() == [0, 1, 2]


def test_reduced_adjacency_overrides_original_edges_with_exactly_two_edges():
    graph = SimpleNamespace(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        feat_syn=torch.ones(2, 2),
        labels_syn=torch.tensor([0, 1]),
        adj_syn=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    data = to_pyg_data(graph)
    assert torch.equal(data.edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert data.num_nodes == 2


@pytest.mark.parametrize("layout", ("dense", "coo", "csr", "scipy", "torch_sparse"))
def test_reduced_weights_and_isolates_survive(layout):
    from scipy.sparse import csr_matrix
    from torch_sparse import SparseTensor

    dense = torch.tensor([[0.0, 0.02, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
    adjacency = {
        "dense": lambda: dense,
        "coo": dense.to_sparse,
        "csr": dense.to_sparse_csr,
        "scipy": lambda: csr_matrix(dense.numpy()),
        "torch_sparse": lambda: SparseTensor.from_dense(dense),
    }[layout]()
    graph = SimpleNamespace(
        feat_syn=torch.ones(3, 2),
        labels_syn=torch.arange(3),
        adj_syn=adjacency,
        train_mask=torch.ones(99, dtype=torch.bool),
    )
    data = to_pyg_data(graph)
    assert data.num_nodes == 3 and data.edge_index.shape == (2, 2)
    assert torch.allclose(data.edge_weight, torch.tensor([0.02, 0.02]))
    assert getattr(data, "train_mask", None) is None


def test_reset_synthetic_fields_do_not_hide_original_features():
    graph = SimpleNamespace(
        feat_syn=None,
        labels_syn=None,
        adj_syn=None,
        x=torch.ones(4, 2),
        y=torch.arange(4),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        train_mask=torch.tensor([True, False, False, False]),
    )
    data = to_pyg_data(graph)
    assert torch.equal(data.x, graph.x) and torch.equal(data.y, graph.y)
    assert torch.equal(data.train_mask, graph.train_mask) and data.num_nodes == 4


def test_real_dgl_roundtrip_preserves_weights_features_splits_and_isolates():
    pytest.importorskip("dgl")
    from torch_geometric.data import Data

    data = Data(
        x=torch.arange(12).reshape(4, 3).float(),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_weight=torch.tensor([0.2, 0.3]),
        edge_attr=torch.ones(2, 3),
        train_mask=torch.tensor([True, False, False, False]),
        val_mask=torch.tensor([False, True, False, False]),
        test_mask=torch.tensor([False, False, True, True]),
        num_nodes=4,
    )
    restored = to_pyg_data(to_dgl_graph(data))
    for key in data.keys():
        if key == "num_nodes":
            assert restored[key] == data[key]
        else:
            assert torch.equal(restored[key], data[key]), key


def test_heterogeneous_dgl_is_not_silently_flattened():
    dgl = pytest.importorskip("dgl")
    graph = dgl.heterograph({("a", "r", "b"): (torch.tensor([0]), torch.tensor([0]))})
    with pytest.raises(ValueError, match="homogeneous"):
        to_pyg_data(graph)


def test_dataset_converter_preserves_typed_edge_values_and_isolated_nodes():
    dgl = pytest.importorskip("dgl")
    from graphslim.dataset.convertor import from_dgl

    graph = dgl.heterograph(
        {("a", "r", "b"): (torch.tensor([0]), torch.tensor([0]))}, num_nodes_dict={"a": 3, "b": 2}
    )
    graph.edges["r"].data["weight"] = torch.tensor([0.25])
    graph.nodes["a"].data["feat"] = torch.ones(3, 2)
    restored = from_dgl(graph, "fixture", hetero=True)
    assert restored["a"].num_nodes == 3 and restored["b"].num_nodes == 2
    assert torch.equal(restored["a", "r", "b"].weight, torch.tensor([0.25]))
    with pytest.raises(ValueError, match="multiple node types"):
        from_dgl(graph, "fixture", hetero=False)


def test_dataset_homogeneous_converter_preserves_feature_aliases():
    dgl = pytest.importorskip("dgl")
    from graphslim.dataset.convertor import from_dgl

    graph = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=4)
    graph.ndata["feat"] = torch.ones(4, 2)
    graph.ndata["label"] = torch.arange(4)
    graph.edata["weight"] = torch.tensor([0.3])
    restored = from_dgl(graph, "fixture")
    assert restored.num_nodes == 4 and torch.equal(restored.x, graph.ndata["feat"])
    assert torch.equal(restored.y, graph.ndata["label"])
    assert torch.equal(restored.edge_weight, graph.edata["weight"])

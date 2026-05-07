from types import SimpleNamespace

import torch

from graphslim.compat import to_pyg_data


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

import torch
from torch_geometric.data import Data

from graphslim.visualization import draw_graph_pair, to_networkx_graph


def _small_graph():
    return Data(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]), x=torch.ones(3, 2))


def test_to_networkx_graph_converts_pyg_data():
    graph = to_networkx_graph(_small_graph())

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3


def test_draw_graph_pair_writes_output(tmp_path):
    output = tmp_path / "pair.png"

    fig, _ = draw_graph_pair(_small_graph(), _small_graph(), output_path=str(output))

    assert output.exists()
    fig.clear()

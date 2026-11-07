# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Actual graph checkpoint loading, CLI rendering, isolated nodes, and sampling."""

import matplotlib.pyplot as plt
import pytest
import torch
from torch_geometric.data import Data

from graphslim import visualization


@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
@pytest.mark.parametrize("with_features", [False, True])
def test_saved_graph_formats_preserve_edge_direction_and_isolated_nodes(
    tmp_path, layout, with_features
):
    # Exactly two edges exposes the ambiguous 2x2 nonzero-index case.
    adjacency = torch.zeros(4, 4)
    adjacency[0, 2], adjacency[1, 0] = 1, 2
    saved = {
        "dense": lambda: adjacency,
        "coo": adjacency.to_sparse,
        "csr": adjacency.to_sparse_csr,
    }[layout]()
    torch.save(saved, tmp_path / "adj.pt")
    torch.save(torch.ones(4, 2), tmp_path / "feat.pt")
    torch.save(torch.tensor([0, 1, 0, 1]), tmp_path / "label.pt")
    result = visualization.load_reduced_graph(
        str(tmp_path / "adj.pt"),
        str(tmp_path / "feat.pt") if with_features else None,
        str(tmp_path / "label.pt") if with_features else None,
    )
    assert result.edge_index.tolist() == [[0, 1], [2, 0]]
    assert result.num_nodes == 4
    if with_features:
        torch.testing.assert_close(result.x, torch.ones(4, 2))
        assert result.y.tolist() == [0, 1, 0, 1]


def test_visualization_cli_creates_image_from_dense_checkpoint(tmp_path):
    torch.save(torch.eye(4), tmp_path / "adj.pt")
    output = tmp_path / "nested/graph.png"
    try:
        visualization.main(["--adj", str(tmp_path / "adj.pt"), "--out", str(output)])
        pixels = plt.imread(output)
        assert pixels.ndim == 3 and pixels.shape[0] > 100
    finally:
        plt.close("all")


def test_invalid_serialized_adjacency_is_rejected(tmp_path):
    torch.save([1, 2, 3], tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="Unsupported adjacency"):
        visualization.load_reduced_graph(str(tmp_path / "invalid.pt"))


def test_feature_only_graphs_and_empty_graphs_render_without_adding_edges(tmp_path):
    feature_only = Data(x=torch.ones(8, 2))
    assert visualization.to_networkx_graph(feature_only).number_of_nodes() == 8
    empty = Data(num_nodes=0)
    fig, axes = visualization.draw_graph_pair(feature_only, empty, max_nodes=3)
    try:
        assert len(axes[0].collections[0].get_offsets()) == 3
        assert len(axes) == 2
    finally:
        plt.close(fig)


def test_visualization_cli_creates_image_from_sparse_checkpoint(tmp_path):
    torch.save(torch.eye(4).to_sparse(), tmp_path / "adj.pt")
    output = tmp_path / "nested/graph.png"
    try:
        visualization.main(["--adj", str(tmp_path / "adj.pt"), "--out", str(output)])
        pixels = plt.imread(output)
        assert pixels.ndim == 3 and pixels.shape[0] > 100
    finally:
        plt.close("all")

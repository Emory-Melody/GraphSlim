"""Exercise native graph interchange with actual node/edge buffers."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset.convertor import (
    csr2ei,
    dense2sparsetensor,
    ei2csr,
    pyg_to_networkit,
)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("weighted", [False, True])
def test_networkit_conversion_keeps_isolates_and_deduplicates_undirected_edges(dtype, weighted):
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=dtype)
    weights = torch.tensor([0.25, 0.25, 2.0, 2.0]) if weighted else None
    graph = pyg_to_networkit(Data(edge_index=edges, edge_attr=weights, num_nodes=5))
    assert graph.numberOfNodes() == 5 and graph.numberOfEdges() == 2
    assert graph.degree(3) == graph.degree(4) == 0
    assert graph.isWeighted() == weighted
    assert graph.weight(0, 1) == pytest.approx(0.25 if weighted else 1)
    # networkit_to_pyg is not exercised: it calls Graph.edges(), which NetworKit 11
    # no longer provides (package bug), so inspect the native graph directly.
    assert {tuple(sorted(pair)) for pair in graph.iterEdges()} == {(0, 1), (1, 2)}
    if weighted:
        assert graph.weight(1, 2) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "edges",
    [
        torch.tensor([0, 1]),
        torch.tensor([[0], [-1]]),
        torch.tensor([[0], [3]]),
        torch.tensor([[0.0], [1.0]]),
    ],
)
def test_invalid_node_buffers_are_rejected_before_native_code(edges):
    with pytest.raises(ValueError, match="edge_index"):
        pyg_to_networkit(Data(edge_index=edges, num_nodes=3))


def test_invalid_edge_weight_shape_is_rejected():
    graph = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 3), num_nodes=2)
    with pytest.raises(ValueError, match="one scalar weight"):
        pyg_to_networkit(graph)


def test_empty_graph_and_sparse_representations_keep_shape():
    graph = pyg_to_networkit(Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=4))
    assert graph.numberOfNodes() == 4 and graph.numberOfEdges() == 0
    dense = torch.tensor([[0.0, 0.25, 0], [2, 0, 0], [0, 0, 0]])
    sparse = dense2sparsetensor(dense)
    torch.testing.assert_close(sparse.to_dense(), dense)
    binary = dense2sparsetensor(dense, has_value=False).to_dense()
    torch.testing.assert_close(binary, (dense != 0).float())
    adjacency = ei2csr(dense.nonzero().T, 3)
    roundtrip = ei2csr(csr2ei(adjacency), 3)
    np.testing.assert_array_equal(roundtrip.toarray(), adjacency.toarray())

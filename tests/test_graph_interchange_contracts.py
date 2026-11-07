# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Exercise native graph interchange with actual node/edge buffers."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset.convertor import (
    csr2ei,
    dense2sparsetensor,
    ei2csr,
    networkit_to_pyg,
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
    # Regression: networkit_to_pyg called Graph.edges(), absent in NetworKit 11.
    restored = networkit_to_pyg(graph)
    assert restored.num_nodes == 5 and restored.edge_index.shape == (2, 2)
    assert {tuple(sorted(pair)) for pair in restored.edge_index.T.tolist()} == {(0, 1), (1, 2)}
    if weighted:
        weights = dict(zip(map(tuple, restored.edge_index.T.tolist()), restored.edge_attr.tolist()))
        assert sorted(weights.values()) == [0.25, 2.0]
    else:
        assert restored.edge_attr is None


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
    restored = networkit_to_pyg(graph)
    assert restored.num_nodes == 4 and restored.edge_index.shape == (2, 0)
    dense = torch.tensor([[0.0, 0.25, 0], [2, 0, 0], [0, 0, 0]])
    sparse = dense2sparsetensor(dense)
    torch.testing.assert_close(sparse.to_dense(), dense)
    binary = dense2sparsetensor(dense, has_value=False).to_dense()
    torch.testing.assert_close(binary, (dense != 0).float())
    adjacency = ei2csr(dense.nonzero().T, 3)
    roundtrip = ei2csr(csr2ei(adjacency), 3)
    np.testing.assert_array_equal(roundtrip.toarray(), adjacency.toarray())

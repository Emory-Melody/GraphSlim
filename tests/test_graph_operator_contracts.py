"""Representation invariance and closed-form checks for graph operators."""

import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from graphslim import utils


@pytest.mark.parametrize("flow", ["source_to_target", "target_to_source"])
@pytest.mark.parametrize("self_loops", [False, True])
@pytest.mark.parametrize("improved", [False, True])
def test_edge_index_normalization_matches_dense_directed_graph(flow, self_loops, improved):
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    weights = torch.tensor([0.5, 1.5, 2.0, 3.0])
    edges, normalized = utils.gcn_norm(
        edge_index, weights, num_nodes=4, improved=improved, add_self_loops=self_loops, flow=flow
    )
    matrix = torch.zeros(4, 4)
    matrix[edge_index[0], edge_index[1]] = weights
    if self_loops:
        matrix += torch.eye(4) * (2 if improved else 1)
    degree = matrix.sum(0 if flow == "source_to_target" else 1)
    inverse = torch.zeros_like(degree)
    inverse[degree > 0] = degree[degree > 0].rsqrt()
    expected = torch.diag(inverse) @ matrix @ torch.diag(inverse)
    actual = torch.zeros_like(matrix)
    actual[edges[0], edges[1]] = normalized
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(normalized).all()


@pytest.mark.parametrize("valued", [False, True])
@pytest.mark.parametrize("self_loops", [False, True])
def test_sparse_normalization_matches_edge_index_and_retains_isolated_nodes(valued, self_loops):
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    values = torch.tensor([0.5, 1.5, 2.0, 3.0]) if valued else None
    adjacency = SparseTensor.from_edge_index(edge_index, values, sparse_sizes=(4, 4)).t()
    normalized = utils.gcn_norm(adjacency, add_self_loops=self_loops)
    edges, weights = utils.gcn_norm(edge_index, values, num_nodes=4, add_self_loops=self_loops)
    expected = SparseTensor.from_edge_index(edges, weights, sparse_sizes=(4, 4)).t()
    torch.testing.assert_close(normalized.to_dense(), expected.to_dense())


def test_sgformer_normalization_agrees_on_binary_graphs_and_drops_isolated_rows():
    adjacency = torch.tensor([[0.0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    dense_result = utils.normalize_adj_sgformer(adjacency)
    sparse_result = utils.normalize_adj_sgformer(SparseTensor.from_dense(adjacency)).to_dense()
    torch.testing.assert_close(sparse_result, dense_result)
    assert torch.isfinite(dense_result).all()
    assert dense_result[3].count_nonzero() == 0
    assert dense_result[0, 1].item() == pytest.approx(1 / np.sqrt(2))


def test_sgformer_sparse_path_uses_edge_counts_for_weighted_graphs():
    # Documented contract: the sparse path normalizes by column edge counts while
    # the dense path uses weighted degrees. A valueless SparseTensor is not
    # supported (stored values are required).
    adjacency = torch.tensor([[0.0, 0.5, 0], [0.5, 0, 2], [0, 2, 0]])
    sparse = utils.normalize_adj_sgformer(SparseTensor.from_dense(adjacency)).to_dense()
    counts = (adjacency != 0).sum(0).float()
    expected = adjacency / torch.sqrt(counts[:, None] * counts[None, :])
    torch.testing.assert_close(sparse, expected)
    dense = utils.normalize_adj_sgformer(adjacency)
    weighted = adjacency.sum(1)
    torch.testing.assert_close(dense, adjacency / torch.sqrt(weighted[:, None] * weighted[None, :]))


def test_feature_regularization_matches_edge_dirichlet_energy_and_has_finite_gradients():
    adjacency = torch.tensor([[0.0, 2, 1], [2, 0, 3], [1, 3, 0]], requires_grad=True)
    features = torch.tensor([[1.0, 3], [2, -1], [4, 2]], requires_grad=True)
    scaled = features.detach().numpy() / np.sqrt(adjacency.detach().numpy().sum(1))[:, None]
    expected = 0.5 * np.sum(
        adjacency.detach().numpy()[:, :, None] * (scaled[:, None] - scaled[None, :]) ** 2
    )
    actual = utils.regularization(adjacency, features)
    assert actual.item() == pytest.approx(expected, rel=1e-6)
    actual.backward()
    assert torch.isfinite(features.grad).all() and torch.isfinite(adjacency.grad).all()
    assert utils.maxdegree(torch.zeros(3, 3)) == 0
    assert utils.sparsity(torch.zeros(3, 3)) == 0
    assert utils.sparsity2(torch.ones(3, 3)).item() == pytest.approx(1)


@pytest.mark.parametrize("center", [False, True])
def test_class_encoding_has_expected_centering_and_pairwise_class_distances(center):
    labels = np.array([0, 1, 2, 0])
    encoded = utils.one_hot_sfgc(labels, 3, center=center)
    np.testing.assert_allclose(encoded.sum(1), np.full(4, 0 if center else 1), atol=1e-7)
    assert encoded.argmax(1).tolist() == labels.tolist()
    np.testing.assert_array_equal(encoded[0], encoded[3])
    assert np.linalg.norm(encoded[0] - encoded[1]) == pytest.approx(np.sqrt(2))

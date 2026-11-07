# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Independent numerical checks, degenerate inputs, and split invariants."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

from graphslim import utils
from graphslim.condensation import utils as condensation


@pytest.mark.parametrize("kind", ["numpy", "tensor", "scipy", "coo", "torch_sparse"])
def test_normalization_matches_closed_form_with_isolated_node(kind):
    matrix = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.float32)
    dense = torch.tensor(matrix)
    expected = np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0], [0, 0, 1]])
    if kind == "numpy":
        actual = utils.gcn_normalize_adj(matrix)
    elif kind == "scipy":
        actual = utils.gcn_normalize_adj(sp.csr_matrix(matrix)).toarray()
    elif kind == "tensor":
        actual = utils.normalize_adj_tensor(dense).numpy()
    elif kind == "coo":
        actual = utils.normalize_sparse_tensor(dense.to_sparse()).to_dense().numpy()
    else:
        actual = utils.gcn_norm(SparseTensor.from_dense(dense)).to_dense().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("improved", [False, True])
@pytest.mark.parametrize("self_loops", [False, True])
def test_gcn_normalization_agrees_for_edge_index_and_sparse_tensor(improved, self_loops):
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    weights = torch.tensor([1.0, 1.0, 2.0, 2.0])
    index, values = utils.gcn_norm(edges, weights, 4, improved, self_loops)
    actual = torch.sparse_coo_tensor(index, values, (4, 4)).to_dense()
    sparse = SparseTensor(row=edges[0], col=edges[1], value=weights, sparse_sizes=(4, 4))
    expected = utils.gcn_norm(sparse, improved=improved, add_self_loops=self_loops).to_dense()
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_batched_and_row_normalization():
    adjacency = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    batch = torch.stack([adjacency, adjacency * 3])
    normalized = utils.normalize_adj_tensor(batch)
    for i in range(2):
        torch.testing.assert_close(normalized[i], utils.dense_gcn_norm(batch[i], "cpu"))
    features = np.array([[2.0, 2.0], [0.0, 0.0], [1.0, 3.0]])
    for value in [features, sp.csr_matrix(features)]:
        actual = utils.normalize_feature(value)
        if sp.issparse(actual):
            actual = actual.toarray()
        np.testing.assert_allclose(actual, [[0.5, 0.5], [0, 0], [0.25, 0.75]])
    for sparse in [False, True]:
        value = adjacency.to_sparse() if sparse else adjacency
        torch.testing.assert_close(
            utils.degree_normalize_adj_tensor(value, sparse=sparse).to_dense().sum(1), torch.ones(2)
        )


def test_sparse_conversion_keeps_weights_isolates_and_integer_labels():
    original = sp.csr_matrix(([0.25, 2.0], ([0, 1], [1, 0])), shape=(4, 4))
    tensor = utils.to_tensor(original)
    np.testing.assert_allclose(utils.to_scipy(tensor).toarray(), original.toarray())
    np.testing.assert_allclose(utils.to_scipy(tensor.to_dense()).toarray(), original.toarray())
    features, labels = utils.to_tensor(np.ones((4, 2)), label=np.array([0, 1, 1, 0]))
    assert features.dtype == torch.float32 and labels.dtype == torch.long
    for value in [torch.eye(3), torch.eye(3).to_sparse(), SparseTensor.eye(3)]:
        assert utils.is_identity(value, "cpu")
    assert not utils.is_identity(torch.zeros(3, 3), "cpu")
    for encoded in [
        utils.encode_onehot(labels.numpy()),
        utils.tensor2onehot(labels),
        utils.one_hot(labels, 2),
    ]:
        np.testing.assert_array_equal(np.asarray(encoded), np.eye(2)[labels])


@pytest.mark.parametrize("metric", ["mse", "cos", "ours"])
def test_gradient_matching_penalizes_mismatch_and_has_finite_derivatives(metric):
    real = [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
    synthetic = [torch.tensor([[2.0, -1.0], [0.5, 1.0]], requires_grad=True)]
    args = SimpleNamespace(dis_metric=metric)
    matching = condensation.match_loss(synthetic, real, args, "cpu")
    equal = condensation.match_loss(real, real, args, "cpu")
    assert matching > equal
    matching.backward()
    assert torch.isfinite(synthetic[0].grad).all() and synthetic[0].grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3), (2, 2, 2, 2)])
def test_weight_distance_handles_bias_linear_and_convolution_shapes(shape):
    weights = torch.arange(1, np.prod(shape) + 1).float().reshape(shape)
    value = condensation.distance_wb(weights, weights)
    assert abs(float(value)) < 1e-4


@pytest.mark.parametrize("scheduler", ["linear", "root", "geom"])
def test_curriculum_starts_at_fraction_and_saturates(scheduler):
    values = [condensation.training_scheduler(0.2, i, 10, scheduler) for i in range(21)]
    assert values[0] == pytest.approx(0.2)
    assert values[10:] == pytest.approx([1] * 11)
    assert values == sorted(values)


def test_spectral_feature_projections_reconstruct_and_cache(tmp_path, monkeypatch):
    # get_eigh caches under ./data/<name>; run it inside tmp_path.
    monkeypatch.chdir(tmp_path)
    adjacency = sp.csr_matrix([[0.0, 1, 0], [1, 0, 1], [0, 1, 0]])
    laplacian = sp.csgraph.laplacian(adjacency)
    values, vectors = condensation.get_eigh(laplacian, "tiny")
    assert (tmp_path / "data/tiny/eigenvalues.npy").exists()
    again = condensation.get_eigh(laplacian, "tiny")
    loaded = condensation.load_eigen("tiny", tmp_path / "data")
    for result in [again, loaded]:
        np.testing.assert_allclose(result[0], values)
        np.testing.assert_allclose(
            result[1] @ np.diag(result[0]) @ result[1].T, laplacian.toarray(), atol=1e-12
        )
    u = torch.tensor(vectors).float()
    x = torch.tensor([[1.0, 2], [3, 1], [0, 5]])
    transformed, pieces = condensation.get_subspace_embed(u, x)
    torch.testing.assert_close(u @ transformed, x)
    torch.testing.assert_close(pieces.sum(0), x)
    covariance = condensation.get_subspace_covariance_matrix(u, x)
    torch.testing.assert_close(covariance, covariance.transpose(1, 2))
    assert torch.linalg.eigvalsh(covariance).min() > -1e-6
    torch.testing.assert_close(condensation.get_embed_sum(torch.zeros(3), u, x), x)
    labels = torch.tensor([0, 0, 1])
    a = condensation.get_embed_mean(x, labels)
    b = condensation.get_embed_mean(x, labels, torch.nn.functional.one_hot(labels).float().T)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a.norm(dim=1), torch.ones(2))


def test_feature_standardization_and_graph_filter():
    x = torch.tensor([[2.0, 1], [2, 3], [2, 5]])
    normalized = condensation.normalize_data(x)
    torch.testing.assert_close(normalized[:, 0], torch.zeros(3))
    assert normalized[:, 1].mean().item() == pytest.approx(0)
    assert normalized[:, 1].std().item() == pytest.approx(1)
    torch.testing.assert_close(condensation.GCF(torch.eye(3), x, k=3), x)
    adjacency = torch.ones(3, 3)
    torch.testing.assert_close(condensation.GCF(adjacency, x), x.mean(0).expand_as(x))


# get_train_val_test_gcn and get_splits_each_class are left out: they call the
# removed ``np.int`` alias (graphslim/utils.py:1036-1037 and 1074-1076).
@pytest.mark.parametrize("splitter", ["three", "two"])
def test_stratified_splits_are_reproducible_disjoint_and_cover_every_class(splitter):
    labels = np.repeat(np.arange(3), 600)
    if splitter == "three":

        def make():
            return utils.get_train_val_test(len(labels), stratify=labels, seed=9)

    else:

        def make():
            return utils.get_train_test(len(labels), stratify=labels, seed=9)

    first, second = make(), make()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert len(set(a)) == len(a)
        assert set(labels[a]) == {0, 1, 2}
    flattened = np.concatenate(first)
    assert len(flattened) == len(set(flattened))


def test_metrics_use_hard_or_onehot_labels_consistently():
    labels = torch.tensor([0, 1, 1, 0])
    scores = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]])
    assert utils.accuracy(scores, labels) == pytest.approx(0.75)
    assert utils.accuracy(scores, labels.numpy()) == pytest.approx(0.75)
    assert utils.f1_macro(scores, labels) == pytest.approx(11 / 15)
    assert utils.roc_auc(scores, labels) == pytest.approx(0.75)
    assert utils.classification_margin(scores[0].log(), 0) == pytest.approx(0.6)

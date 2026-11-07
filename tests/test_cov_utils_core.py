# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Closed-form checks for the remaining public helpers in ``graphslim.utils``."""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from graphslim import utils


def path_graph(n=4):
    adjacency = torch.zeros(n, n)
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    return adjacency


def test_is_identity_rejects_unsupported_inputs_and_detects_identity_sparse_forms():
    class MatrixLike:
        def size(self, dim):
            return 2

    with pytest.raises(ValueError, match="torch.Tensor"):
        utils.is_identity(MatrixLike(), "cpu")
    assert utils.is_identity(SparseTensor.eye(2), "cpu")
    assert not utils.is_identity(SparseTensor.from_dense(path_graph(2)), "cpu")


def test_seed_everything_makes_all_generators_reproducible(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    utils.seed_everything(13)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    assert os.environ["PYTHONHASHSEED"] == "13"
    utils.seed_everything(13)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first


def test_row_normalize_tensor_shifts_by_global_minimum_in_place():
    matrix = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    result = utils.row_normalize_tensor(matrix)
    torch.testing.assert_close(result, torch.tensor([[0.0, 1.0], [1 / 3, 2 / 3]]))
    torch.testing.assert_close(result.sum(1), torch.ones(2))
    # The minimum subtraction is documented to mutate the argument.
    torch.testing.assert_close(matrix, torch.tensor([[0.0, 1.0], [2.0, 4.0]]))


def test_calc_delegates_all_arguments_to_kernel_object():
    calls = []

    class Kernel:
        def gntk(self, *args):
            calls.append(args)
            return 7.5

    assert utils.calc(Kernel(), 1, 2, 3, 4, 5, 6) == 7.5
    assert calls == [(1, 2, 3, 4, 5, 6)]


def test_index_mask_roundtrip_and_camel_case():
    mask = utils.index_to_mask(torch.tensor([0, 3]), 5)
    assert mask.dtype == torch.bool and mask.tolist() == [True, False, False, True, False]
    np.testing.assert_array_equal(utils.mask_to_index(mask.numpy(), 5), [0, 3])
    np.testing.assert_array_equal(utils.mask_to_index([4, 1], 5), [4, 1])
    assert utils.to_camel_case("variation_neighborhoods") == "VariationNeighborhoods"
    assert utils.to_camel_case("gcond") == "Gcond"


def test_to_tensor_orders_positional_then_keyword_values_and_casts_labels():
    features = np.arange(6, dtype=np.float64).reshape(3, 2)
    adjacency = sp.csr_matrix(np.eye(3))
    feat, adj, label, weights, labels_2d = utils.to_tensor(
        features,
        adjacency,
        label=np.array([2, 0, 1]),
        weights=torch.tensor([1, 2, 3]),
        label_onehot=np.eye(3),
        device="cpu",
    )
    assert feat.dtype == torch.float32 and feat.tolist() == features.tolist()
    assert adj.is_sparse and torch.equal(adj.to_dense(), torch.eye(3))
    assert label.dtype == torch.long and label.tolist() == [2, 0, 1]
    # Non-label keyword values and multi-dimensional "label" values stay float.
    assert weights.dtype == torch.float32 and weights.tolist() == [1.0, 2.0, 3.0]
    assert labels_2d.dtype == torch.float32 and labels_2d.shape == (3, 3)
    single = utils.to_tensor(torch.tensor([1, 2]))
    assert isinstance(single, torch.Tensor) and single.dtype == torch.float32


def test_normalize_feature_accepts_lil_matrix_and_keeps_zero_rows():
    matrix = sp.lil_matrix(np.array([[1.0, 3.0], [0.0, 0.0]]))
    result = utils.normalize_feature(matrix)
    np.testing.assert_allclose(result.toarray(), [[0.25, 0.75], [0.0, 0.0]])


@pytest.mark.parametrize("weighted", [False, True])
def test_add_self_loops_appends_loops_with_fill_value(weighted):
    edges = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([2.0, 2.0]) if weighted else None
    index, values = utils.add_self_loops(edges, weights, fill_value=5, num_nodes=3)
    assert index.tolist() == [[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]
    if weighted:
        assert values.tolist() == [2.0, 2.0, 5.0, 5.0, 5.0]
    else:
        assert values is None


def test_sparse_gcn_normalization_returns_transposed_torch_sparse_matrix():
    adjacency = torch.tensor([[0.0, 1, 1], [1, 0, 0], [1, 0, 0]])
    result = utils.normalize_adj_tensor(adjacency.to_sparse(), sparse=True)
    assert isinstance(result, SparseTensor)
    expected = utils.dense_gcn_norm(adjacency, "cpu")
    torch.testing.assert_close(result.to_dense(), expected.t())
    # The dense path also accepts NumPy input and converts it to a tensor.
    numpy_result = utils.dense_gcn_norm(adjacency.numpy(), "cpu")
    torch.testing.assert_close(numpy_result, expected)
    torch.testing.assert_close(utils.gcn_normalize_adj(adjacency), expected)


def test_coo_degree_normalizations_match_dense_closed_forms():
    adjacency = torch.tensor([[0.0, 1, 1], [1, 0, 0], [1, 0, 0]])
    looped = adjacency + torch.eye(3)
    degree = looped.sum(1)
    symmetric = utils.normalize_sparse_tensor(adjacency.to_sparse()).to_dense()
    torch.testing.assert_close(symmetric, looped / torch.sqrt(degree[:, None] * degree[None, :]))
    row = utils.degree_normalize_sparse_tensor(adjacency.to_sparse()).to_dense()
    torch.testing.assert_close(row, looped / degree[:, None])
    torch.testing.assert_close(row.sum(1), torch.ones(3))


def test_scipy_degree_normalization_adds_identity_only_for_empty_first_diagonal():
    without_loops = sp.csr_matrix(np.array([[0.0, 1], [1, 0]]))
    np.testing.assert_allclose(
        utils.degree_normalize_adj(without_loops).toarray(), [[0.5, 0.5], [0.5, 0.5]]
    )
    with_loops = sp.csr_matrix(np.array([[1.0, 3], [1, 0]]))
    np.testing.assert_allclose(
        utils.degree_normalize_adj(with_loops).toarray(), [[0.25, 0.75], [1.0, 0.0]]
    )


def test_metrics_wrap_scalar_and_array_labels():
    output = torch.tensor([[0.2, 0.8]])
    assert utils.accuracy(output, np.array([1])) == 1
    assert utils.accuracy(output, 0) == 0
    scores = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
    assert utils.roc_auc(scores, np.array([0, 1])) == 1
    assert utils.f1_macro(scores, np.array([0, 1])) == 1
    assert utils.f1_macro(output, 1) == 1
    # A single scalar label is wrapped into a one-element sample; AUC is undefined.
    with pytest.warns(UserWarning, match="Only one class"):
        assert np.isnan(utils.roc_auc(output, 1))


def test_classification_margin_from_log_probabilities():
    log_probs = torch.log(torch.tensor([0.1, 0.6, 0.3]))
    assert utils.classification_margin(log_probs, 1) == pytest.approx(0.3)
    assert utils.classification_margin(log_probs, 0) == pytest.approx(-0.5)


def test_split_helpers_require_stratification_labels():
    with pytest.raises(AssertionError, match="stratify"):
        utils.get_train_val_test(10)
    with pytest.raises(AssertionError, match="stratify"):
        utils.get_train_test(10)


def test_unseeded_splits_have_requested_sizes():
    labels = np.repeat([0, 1], 50)
    train, val, test = utils.get_train_val_test(100, 0.2, 0.6, stratify=labels)
    assert (len(train), len(val), len(test)) == (20, 20, 60)
    assert len(set(train) | set(val) | set(test)) == 100
    train, test = utils.get_train_test(100, 0.7, stratify=labels)
    assert (len(train), len(test)) == (30, 70)
    np.testing.assert_array_equal(np.bincount(labels[train]), [15, 15])


def test_degree_sequence_from_dense_and_sparse_adjacency():
    adjacency = path_graph()
    torch.testing.assert_close(utils.get_degree_squence(adjacency), torch.tensor([1.0, 2, 2, 1]))
    sparse = utils.get_degree_squence(adjacency.to_sparse()).to_dense()
    torch.testing.assert_close(sparse, torch.tensor([1.0, 2, 2, 1]))


def expected_powerlaw(degrees, d_min):
    kept = degrees[degrees >= d_min]
    n, total = len(kept), np.log(kept).sum()
    alpha = 1 + n / (total - n * np.log(d_min - 0.5))
    ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * total
    return ll, alpha, n, total


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_degree_log_likelihood_matches_closed_form_and_ignores_low_degrees(backend):
    degrees = np.array([0.0, 1, 2, 3, 4, 7])
    expected = expected_powerlaw(degrees, 2.0)
    if backend == "numpy":
        actual = utils.degree_sequence_log_likelihood(degrees, np.float64(2.0))
    else:
        actual = utils.degree_sequence_log_likelihood(
            torch.tensor(degrees), torch.tensor(2.0, dtype=torch.float64)
        )
    assert actual[2] == expected[2] == 4
    for value, target in zip(actual, expected):
        assert float(value) == pytest.approx(target)


def test_power_law_helpers_accept_python_scalars():
    alpha = utils.compute_alpha(3, 4.0, 2.0)
    assert alpha == pytest.approx(1 + 3 / (4.0 - 3 * np.log(1.5)))
    ll = utils.compute_log_likelihood(3, alpha, 4.0, 2.0)
    assert ll == pytest.approx(3 * np.log(alpha) + 3 * alpha * np.log(2.0) + (alpha + 1) * 4.0)


def test_incremental_sum_log_degrees_matches_recomputation():
    # updated_log_likelihood_for_edge_changes itself is not exercised: it indexes
    # ``adjacency_matrix[node_pairs.T]`` (graphslim/utils.py:1140), which selects
    # whole rows instead of the candidate entries and fails for every input.
    degrees = torch.tensor([1.0, 3.0, 2.0, 5.0])
    d_min = torch.tensor(2.0)
    kept = degrees[degrees >= d_min]
    before = torch.log(kept).sum()
    pairs = torch.tensor([[0, 1], [2, 3]])
    deltas = torch.tensor([1.0, -1.0])
    old = degrees[pairs]
    new = old + deltas[:, None]
    total, count = utils.update_sum_log_degrees(before, len(kept), old, new, d_min)
    for i in range(2):
        changed = degrees.clone()
        changed[pairs[i]] += deltas[i]
        filtered = changed[changed >= d_min]
        assert total[i].item() == pytest.approx(torch.log(filtered).sum().item())
        assert count[i].item() == len(filtered)
    assert count.dtype == torch.float32


def test_feature_noise_replaces_requested_fraction_and_reports_delta():
    original = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    data = Data(x=original.clone())
    delta, mask = utils.add_feature_noise(data, 0.3, seed=1)
    assert mask.dtype == torch.bool and int(mask.sum()) == 3
    torch.testing.assert_close(data.x[~mask], original[~mask])
    torch.testing.assert_close(original + delta, data.x)
    assert torch.equal(delta[~mask], torch.zeros_like(delta[~mask]))
    repeat = Data(x=original.clone())
    utils.add_feature_noise(repeat, 0.3, seed=1)
    torch.testing.assert_close(repeat.x, data.x)


def test_test_feature_noise_only_touches_test_nodes():
    original = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    test_mask = torch.tensor([False] * 4 + [True] * 6)
    data = Data(x=original.clone(), test_mask=test_mask)
    delta, mask = utils.add_feature_noise_test(data, 0.5, seed=2)
    assert int(mask.sum()) == 3
    assert not mask[~test_mask].any()
    torch.testing.assert_close(data.x[:4], original[:4])
    torch.testing.assert_close(original + delta, data.x)


def test_add_mask_uses_dataset_split_and_squeezes_labels():
    data = Data(x=torch.ones(4, 1), y=torch.tensor([[0], [1], [1], [0]]))
    split = {"train": [0], "valid": [1, 2], "test": [3]}
    utils.add_mask(data, SimpleNamespace(get_idx_split=lambda: split))
    assert data.train_mask.tolist() == [True, False, False, False]
    assert data.val_mask.tolist() == [False, True, True, False]
    assert data.test_mask.tolist() == [False, False, False, True]
    assert data.y.tolist() == [0, 1, 1, 0]

"""Independent numerical answers for public metrics and graph utilities."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from graphslim import utils
from graphslim.evaluation import Evaluator
from test_lightweight_methods import quick_args


# Live (requires_grad) outputs and one-dimensional scores are not accepted by
# roc_auc/f1_macro (they call ``.numpy()`` without detaching), and
# f1_macro(is_sigmoid=True) thresholds the caller's tensor in place
# (graphslim/utils.py:829-831); those cases are intentionally not asserted here.
@pytest.mark.parametrize("sigmoid", [False, True])
def test_binary_auc_uses_positive_column_without_mutating_predictions(sigmoid):
    scores = torch.tensor([0.1, 0.8, 0.2, 0.9])
    output = torch.stack([1 - scores, scores], 1)
    original = output.clone()
    assert utils.roc_auc(output, [0, 1, 0, 1], is_sigmoid=sigmoid) == 1
    torch.testing.assert_close(output, original)
    reversed_output = output.flip(1)
    assert utils.roc_auc(reversed_output, torch.tensor([0, 1, 0, 1]), sigmoid) == 0


def test_single_column_auc_degrades_to_constant_argmax_prediction():
    output = torch.tensor([[0.1], [0.8], [0.2], [0.9]])
    assert utils.roc_auc(output, [0, 1, 0, 1]) == pytest.approx(0.5)


def test_f1_metric_retains_input_and_scalar_labels():
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    original = output.clone()
    assert utils.f1_macro(output, [0, 1, 0, 1]) == 1
    assert utils.f1_macro(output, torch.tensor([1, 1, 0, 1])) == pytest.approx(11 / 15)
    torch.testing.assert_close(output, original)
    assert utils.accuracy(torch.tensor([[0.1, 0.9]]), 1) == 1


def test_f1_sigmoid_thresholds_positive_column_on_a_copy():
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.6], [0.1, 0.4]])
    assert utils.f1_macro(output.clone(), [0, 1, 1, 0], is_sigmoid=True) == 1


@pytest.mark.parametrize("reg", [0, 0.2])
def test_kernel_evaluation_matches_closed_form_identity_prototypes(reg):
    loss, accuracy = utils.loss_acc_fn_eval(None, np.eye(2), np.eye(2), np.eye(2), np.eye(2), reg)
    assert loss == pytest.approx(reg**2 / (4 * (1 + reg) ** 2))
    assert accuracy == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_degree_normalization_matches_closed_form_random_walk(sparse):
    matrix = torch.tensor([[0.0, 2, 0], [2, 0, 3], [0, 3, 0]])
    value = matrix.to_sparse() if sparse else matrix
    normalized = utils.degree_normalize_adj_tensor(value, sparse=sparse).to_dense()
    looped = matrix + torch.eye(3)
    torch.testing.assert_close(normalized, looped / looped.sum(1, keepdim=True))


def test_mask_helpers_and_graph_splits_preserve_official_node_ids():
    raw = Data(x=torch.ones(6, 2), y=torch.tensor([[0], [1], [0], [1], [0], [1]]))
    official = {
        "train": torch.tensor([1, 4]),
        "valid": torch.tensor([0, 3]),
        "test": torch.tensor([2, 5]),
    }
    utils.add_mask(raw, SimpleNamespace(get_idx_split=lambda: official))
    assert utils.mask_to_index(raw.train_mask.numpy(), 6).tolist() == [1, 4]
    assert raw.y.shape == (6,)
    assert torch.stack([raw.train_mask, raw.val_mask, raw.test_mask]).sum(0).eq(1).all()
    torch.testing.assert_close(raw.test_mask, utils.index_to_mask([2, 5], 6))


def test_flattened_indices_roundtrip_for_rectangular_shapes_and_matrix_padding():
    rows, cols = utils.unravel_index(torch.arange(15), (5, 3))
    indices = torch.stack([rows, cols], 1).numpy()
    np.testing.assert_array_equal(utils.ravel_multiple_indices(indices, (5, 3)), np.arange(15))
    np.testing.assert_array_equal(
        utils.ravel_multiple_indices(indices, (3, 5), reverse=True), cols.numpy() * 5 + rows.numpy()
    )
    original = sp.csr_matrix([[1.0, 0], [0, 2]])
    enlarged = utils.reshape_mx(original, (4, 4)).toarray()
    np.testing.assert_array_equal(enlarged[:2, :2], original.toarray())
    assert np.count_nonzero(enlarged) == 2


# get_train_test_labelrate is left out: it reaches the removed ``np.int`` alias
# through get_splits_each_class (graphslim/utils.py:1074-1076).


def test_tsne_visualization_uses_actual_embeddings_and_writes_four_class_groups(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    args = quick_args(tmp_path / "checkpoints", "random", "trans")
    generator = torch.Generator().manual_seed(3)
    features, labels = torch.rand(12, 3, generator=generator), torch.arange(12) % 2
    try:
        Evaluator(args).tsne_vis(features, labels, features[:6], labels[:6])
        files = list(tmp_path.glob("tsne_visualization_*.pdf"))
        assert len(files) == 1 and files[0].read_bytes().startswith(b"%PDF")
        assert [len(value.get_offsets()) for value in plt.gca().collections] == [6, 3, 6, 3]
    finally:
        plt.close("all")

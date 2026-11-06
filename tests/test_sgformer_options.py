"""Real SGFormer configuration, representation, and checkpoint invariants.

Ported from the legacy suite. The layout test was replaced: sparse inputs are
documented to normalize by edge counts, dense inputs by weighted degree.
"""

import numpy as np
import pytest
import torch

from graphslim.models.sgformer import SGFormer, normalize_adj
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("aggregate", ["add", "cat"])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_configurations_are_permutation_equivariant_with_finite_gradients(
    tmp_path, aggregate, use_graph, weighted
):
    args = quick_args(tmp_path, "random", "trans")
    args.trans_dropout, args.trans_num_layers = 0, 2
    torch.manual_seed(3)
    model = SGFormer(
        3,
        4,
        2,
        args,
        aggregate=aggregate,
        use_graph=use_graph,
        trans_num_heads=2,
        trans_use_weight=weighted,
        trans_use_bn=weighted,
        trans_use_act=weighted,
        trans_use_residual=weighted,
        gnn_use_init=weighted,
        gnn_use_weight=weighted,
        gnn_use_bn=weighted,
        gnn_use_residual=weighted,
        gnn_use_act=weighted,
    ).eval()
    x = torch.randn(6, 3, requires_grad=True)
    adjacency = torch.eye(6).roll(1, 0) + torch.eye(6).roll(-1, 0)
    order = torch.tensor([2, 0, 5, 1, 4, 3])
    output = model(x, adjacency)
    reordered = model(x[order], adjacency[order][:, order])
    torch.testing.assert_close(reordered, output[order])
    torch.testing.assert_close(output.exp().sum(1), torch.ones(6))
    attention = model.get_attentions(x)
    assert attention.shape == (2, 6, 6) and torch.isfinite(attention).all()
    torch.testing.assert_close(model.get_attentions(x[order]), attention[:, order][:, :, order])
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_dense_normalization_uses_weighted_degree_and_zeroes_isolated_nodes():
    adjacency = torch.tensor([[0.0, 0.5, 0, 0], [0.5, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    degree = adjacency.sum(1)
    scale = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    expected = scale[:, None] * adjacency * scale[None, :]
    torch.testing.assert_close(normalize_adj(adjacency), expected)


def test_sparse_normalization_uses_edge_counts_and_matches_dense_for_binary_graphs():
    weighted = torch.tensor([[0.0, 0.5, 0, 0], [0.5, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    result = normalize_adj(weighted.to_sparse().coalesce())
    assert result.is_sparse
    counts = (weighted != 0).sum(0).float()
    scale = torch.where(counts > 0, counts.rsqrt(), torch.zeros_like(counts))
    torch.testing.assert_close(result.to_dense(), scale[:, None] * weighted * scale[None, :])
    binary = (weighted != 0).float()
    torch.testing.assert_close(
        normalize_adj(binary.to_sparse().coalesce()).to_dense(), normalize_adj(binary)
    )


def test_zero_validation_score_still_returns_a_reloadable_fitted_model(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.lr = 0
    graph = tiny_graph()
    graph.labels_val = torch.ones_like(graph.labels_val)
    model = SGFormer(8, 4, 2, args)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor([20.0, -20]))
    score = model.fit_with_val(graph, train_iters=1, verbose=True)
    assert score == 0
    assert torch.isfinite(model.predict(graph.feat_full, graph.adj_full)).all()


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("loss", ["MSE", "CE"])
def test_soft_target_fits_return_finite_predictions_and_test_metrics(tmp_path, setting, loss):
    args = quick_args(tmp_path, "random", setting)
    args.eval_loss = loss
    graph = tiny_graph()
    graph.feat_syn = graph.feat_train.clone()
    graph.adj_syn = torch.eye(len(graph.feat_syn))
    graph.labels_syn = torch.nn.functional.one_hot(graph.labels_train, 2).float() * 3
    model = SGFormer(8, 4, 2, args)
    score = model.fit_with_val(graph, train_iters=3, reduced=True, setting=setting, verbose=True)
    assert 0 <= score <= 1
    result = model.test(graph, setting=setting, verbose=True)
    assert np.isfinite(result) and 0 <= result <= 1


def test_invalid_aggregation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="aggregate"):
        SGFormer(3, 4, 2, quick_args(tmp_path, "random", "trans"), aggregate="unknown")

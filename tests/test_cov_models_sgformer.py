"""SGFormer branch options, configuration sources, and training/test reporting."""

import pytest
import torch
import torch.nn.functional as F

from graphslim.models.sgformer import GraphConvLayer, SGFormer, TransConvLayer, normalize_adj
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_graph_layer_with_weight_only_transforms_normalized_propagation():
    torch.manual_seed(0)
    layer = GraphConvLayer(3, 2, use_weight=True, use_init=False)
    x, x0 = torch.randn(4, 3), torch.randn(4, 3)
    adjacency = torch.ones(4, 4) - torch.eye(4)
    expected = layer.W(normalize_adj(adjacency) @ x)
    torch.testing.assert_close(layer(x, adjacency, x0), expected)
    bare = GraphConvLayer(3, 3, use_weight=False)
    torch.testing.assert_close(bare(x, adjacency, x0), normalize_adj(adjacency) @ x)


def test_weighted_attention_resets_value_projection_and_returns_attention():
    torch.manual_seed(1)
    layer = TransConvLayer(3, 2, num_heads=2, use_weight=True)
    before = layer.Wv.weight.clone()
    layer.reset_parameters()
    assert not torch.equal(before, layer.Wv.weight)
    x = torch.randn(5, 3)
    output, attention = layer(x, x, output_attn=True)
    assert output.shape == (5, 2) and attention.shape == (5, 5)
    torch.testing.assert_close(layer(x, x), output)


def test_configuration_values_come_from_args_or_constructor(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.train_weight_decay, args.trans_weight_decay, args.trans_dropout = 1, 0.25, 0.1
    model = SGFormer(3, 4, 2, args)
    assert model.trans_weight_decay == 0.25 and model.trans_dropout == 0.1
    assert model.trans_num_layers == args.trans_num_layers == len(model.trans_conv.convs)
    del args.trans_num_layers, args.trans_dropout, args.train_weight_decay
    model = SGFormer(3, 4, 2, args, trans_num_layers=2, trans_dropout=0.3)
    assert model.trans_num_layers == 2 and len(model.trans_conv.convs) == 2
    assert model.trans_dropout == 0.3 and model.trans_weight_decay == 0.001


def test_reset_without_graph_branch_only_resets_transformer(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    model = SGFormer(3, 4, 2, args, use_graph=False)
    graph_state = {k: v.clone() for k, v in model.graph_conv.state_dict().items()}
    trans_state = model.trans_conv.fcs[0].weight.clone()
    model.reset_parameters()
    for key, value in model.graph_conv.state_dict().items():
        assert torch.equal(value, graph_state[key])
    assert not torch.equal(trans_state, model.trans_conv.fcs[0].weight)


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_hard_label_fit_selects_validation_weights_and_reports_test_metric(
    tmp_path, capsys, setting
):
    args = quick_args(tmp_path, "random", setting)
    graph = tiny_graph()
    torch.manual_seed(2)
    model = SGFormer(8, 4, 2, args)
    score = model.fit_with_val(graph, train_iters=3, setting=setting, verbose=True)
    out = capsys.readouterr().out
    assert "training acc" in out and "picking the best model" in out
    features = graph.feat_val if setting == "ind" else graph.feat_full
    adjacency = graph.adj_val if setting == "ind" else graph.adj_full
    prediction = model.predict(features, adjacency).argmax(1)
    if setting == "trans":
        prediction = prediction[graph.idx_val]
    assert float(score) == pytest.approx((prediction == graph.labels_val).float().mean().item())
    accuracy = model.test(graph, setting=setting, verbose=True)
    assert "Test set results" in capsys.readouterr().out
    assert 0 <= accuracy <= 1
    assert model.test(graph, setting=setting) == accuracy


def test_reduced_hard_labels_use_negative_log_likelihood(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    graph = tiny_graph()
    graph.feat_syn, graph.adj_syn = graph.feat_train[:6], torch.eye(6)
    graph.labels_syn = graph.labels_train[:6]
    model = SGFormer(8, 4, 2, args)
    model.fit_with_val(graph, train_iters=2, reduced=True)
    assert model.loss is F.nll_loss


def test_soft_cross_entropy_matches_softmax_target_weighting(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.eval_loss = "CE"
    graph = tiny_graph()
    graph.feat_syn, graph.adj_syn = graph.feat_train[:6], torch.eye(6)
    graph.labels_syn = torch.randn(6, 2)
    model = SGFormer(8, 4, 2, args)
    model.fit_with_val(graph, train_iters=1, reduced=True)
    output, target = torch.log_softmax(torch.randn(3, 2), 1), torch.randn(3, 2)
    expected = -(target.softmax(1) * output).sum(1).mean()
    torch.testing.assert_close(model.loss(output, target), expected)

"""Downstream GNN constructors, forward algebra, and BaseGNN training branches."""

import pytest
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from graphslim.models import GAT, GCN, GIN, SGC, Cheby, GraphSage, get_model_class
from graphslim.models.layers import ChebConvolution, GATConv, GraphConvolution
from graphslim.utils import normalize_adj_tensor
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def ring(n=6):
    eye = torch.eye(n)
    return eye.roll(1, 0) + eye.roll(-1, 0)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown GNN model"):
        get_model_class("MLP")
    assert get_model_class("GCN") is GCN


def test_gin_requires_a_graph_layer(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.nlayers = 0
    with pytest.raises(ValueError, match="at least one graph layer"):
        GIN(4, 8, 2, args)


@pytest.mark.parametrize("cls", [GCN, Cheby, GraphSage])
def test_deep_models_with_batch_norm_build_hidden_stack_and_normalize(tmp_path, cls):
    args = quick_args(tmp_path, "random", "trans")
    args.nlayers, args.with_bn, args.dropout = 3, True, 0.0
    torch.manual_seed(0)
    model = cls(4, 5, 2, args)
    assert len(model.layers) == 3 and len(model.bns) == 2
    x, adjacency = torch.randn(7, 4), SparseTensor.from_dense(normalize_adj_tensor(ring(7)))
    output = model(x, adjacency)
    assert output.shape == (7, 2)
    torch.testing.assert_close(output.exp().sum(1), torch.ones(7))
    layer_features, same = model(x, adjacency, output_layer_features=True)
    torch.testing.assert_close(same, output)
    assert [tuple(f.shape) for f in layer_features] == [(7, 5), (7, 5), (7, 2)]
    model.eval()
    hidden = model.layers[0](x, adjacency)
    torch.testing.assert_close(
        model.bns[0](hidden),
        (hidden - model.bns[0].running_mean) / (model.bns[0].running_var + 1e-5).sqrt(),
    )
    # Regression: dense (1, N, C) layer outputs are normalized over node rows.
    torch.testing.assert_close(model(x, adjacency.to_dense()), model(x, adjacency))
    model.train()
    torch.testing.assert_close(model(x, adjacency.to_dense()), model(x, adjacency))


@pytest.mark.parametrize("cls", [GCN, Cheby, GraphSage])
def test_single_layer_models_map_features_directly_to_classes(tmp_path, cls):
    args = quick_args(tmp_path, "random", "trans")
    args.nlayers = 1
    model = cls(4, 6, 3, args)
    assert len(model.layers) == 1
    assert model(torch.randn(5, 4), torch.eye(5)).shape == (5, 3)


def test_linear_activation_makes_two_layer_gcn_an_affine_composition(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.activation, args.dropout = "linear", 0.0
    model = GCN(4, 5, 2, args).eval()
    assert model.activation is None and not model.with_relu
    x, adjacency = torch.randn(6, 4), normalize_adj_tensor(ring())
    first, second = model.layers
    hidden = adjacency @ (x @ first.weight) + first.bias
    expected = F.log_softmax(adjacency @ (hidden @ second.weight) + second.bias, dim=1)
    torch.testing.assert_close(model(x, adjacency), expected)


def test_multilabel_models_return_sigmoid_probabilities(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.multi_label = True
    model = GCN(4, 5, 3, args).eval()
    x = torch.randn(6, 4)
    output = model(x, torch.eye(6))
    assert ((output > 0) & (output < 1)).all()
    args.multi_label = False
    logits_model = GCN(4, 5, 3, args).eval()
    logits_model.load_state_dict(model.state_dict())
    # Both heads share logits: logit(sigmoid) - log_softmax is constant per row.
    shift = torch.logit(output) - logits_model(x, torch.eye(6))
    torch.testing.assert_close(shift, shift[:, :1].expand_as(shift), rtol=1e-4, atol=1e-4)


def test_layer_lists_propagate_one_adjacency_per_layer(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.dropout = 0.0
    model = GCN(4, 5, 2, args).eval()
    x, adjacency = torch.randn(6, 4), normalize_adj_tensor(ring())
    identity = torch.eye(6)
    first, second = model.layers
    hidden = torch.relu(adjacency @ (x @ first.weight) + first.bias)
    expected = F.log_softmax(identity @ (hidden @ second.weight) + second.bias, dim=1)
    torch.testing.assert_close(model(x, [adjacency, identity]), expected)


@pytest.mark.parametrize("ntrans", [2, 3])
def test_sgc_feature_transforms_then_propagates_nlayers_hops(tmp_path, ntrans):
    args = quick_args(tmp_path, "random", "trans")
    args.ntrans, args.with_bn, args.dropout, args.nlayers = ntrans, True, 0.0, 2
    model = SGC(4, 5, 2, args).eval()
    assert len(model.layers) == ntrans and len(model.bns) == ntrans - 1
    x, adjacency = torch.randn(6, 4), normalize_adj_tensor(ring())
    dense = model(x, adjacency)
    torch.testing.assert_close(model(x, SparseTensor.from_dense(adjacency)), dense)
    sparse_list = [SparseTensor.from_dense(adjacency)] * 2
    torch.testing.assert_close(model(x, sparse_list), dense)
    hidden, log_probs = model(x, adjacency, output_layer_features=True)
    torch.testing.assert_close(F.log_softmax(hidden, dim=1), log_probs)
    assert SGC(4, 5, 2, args, mode="eval").ntrans == 1


def test_sgc_hops_equal_matrix_power_of_linear_output(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.ntrans, args.nlayers = 1, 3
    model = SGC(4, 5, 2, args).eval()
    x, adjacency = torch.randn(6, 4), normalize_adj_tensor(ring())
    linear = model.layers[0]
    expected = torch.linalg.matrix_power(adjacency, 3) @ (x @ linear.weight + linear.bias)
    torch.testing.assert_close(model(x, adjacency), F.log_softmax(expected, dim=1))


def test_gat_list_adjacency_concatenates_per_graph_predictions(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.dropout = 0.0
    torch.manual_seed(1)
    model = GAT(4, 3, 2, args, mode="eval").eval()
    assert model.heads == 8 and model.output_heads == 1
    x = torch.randn(6, 4)
    first = SparseTensor.from_dense(ring())
    second = SparseTensor.from_dense(torch.eye(6))
    stacked = model(x, [first, second])
    assert stacked.shape == (12, 2)
    torch.testing.assert_close(stacked[:6], model(x, first))
    torch.testing.assert_close(stacked[6:], model(x, second))


def test_gat_layer_without_bias_and_edge_weights_scale_messages():
    torch.manual_seed(3)
    layer = GATConv(3, 2, heads=2, bias=False)
    assert layer.bias is None
    x = torch.randn(4, 3)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    plain = layer(x, edges)
    # Weights align with the edge list after self loops are added.
    weighted = layer(x, edges, edge_weight=torch.full((8,), 2.0))
    torch.testing.assert_close(weighted, 2 * plain)
    # A mismatched cached weight is replaced by the supplied one.
    layer.edge_weight = torch.ones(3)
    torch.testing.assert_close(layer(x, edges, edge_weight=torch.full((8,), 3.0)), 3 * plain)


def test_gat_sparse_attention_weights_are_row_normalized():
    torch.manual_seed(5)
    layer = GATConv(3, 2, heads=2)
    adjacency = SparseTensor.from_dense(ring(5))
    output, attention = layer(torch.randn(5, 3), adjacency, return_attention_weights=True)
    assert output.shape == (5, 4)
    row, _, value = attention.coo()
    assert value.shape == (15, 2)
    sums = torch.zeros(5, 2).index_add_(0, row, value)
    torch.testing.assert_close(sums, torch.ones(5, 2))


def test_glorot_layer_without_bias_and_single_order_chebyshev_is_linear():
    layer = GraphConvolution(3, 2, with_bias=False, weight_initializer="glorot")
    assert layer.bias is None
    x = torch.randn(4, 3)
    torch.testing.assert_close(layer(x, torch.eye(4)).reshape(4, 2), x @ layer.weight)
    cheb = ChebConvolution(3, 2, with_bias=False, K=1)
    torch.testing.assert_close(cheb(x, ring(4)), x @ cheb.lins[0].weight)


def soft_graph(log_targets=True):
    graph = tiny_graph()
    graph.feat_syn = graph.feat_train.clone()
    graph.adj_syn = torch.eye(16)
    probabilities = F.one_hot(graph.labels_train, 2).float() * 0.8 + 0.1
    graph.labels_syn = probabilities.log() if log_targets else probabilities
    return graph


@pytest.mark.parametrize(
    "method,soft_label,eval_loss,expected",
    [
        ("geom", True, "CE", torch.nn.KLDivLoss),
        ("random", False, "MSE", torch.nn.MSELoss),
        ("random", False, "KLD", torch.nn.KLDivLoss),
    ],
)
def test_soft_synthetic_targets_select_matching_losses(
    tmp_path, method, soft_label, eval_loss, expected
):
    args = quick_args(tmp_path, method, "trans")
    args.soft_label, args.eval_loss, args.eval_wd = soft_label, eval_loss, 0.0
    model = GCN(8, 8, 2, args)
    score = model.fit_with_val(soft_graph(eval_loss != "MSE"), train_iters=3, reduced=True)
    assert isinstance(model.loss, expected)
    assert model.weight_decay == (0.0 if method != "geom" else model.weight_decay)
    assert 0 <= score <= 1


def test_gcgp_mean_squared_loss_compares_probabilities(tmp_path):
    args = quick_args(tmp_path, "gcgp", "trans")
    args.eval_loss = "MSE"
    model = GCN(8, 8, 2, args)
    model.fit_with_val(soft_graph(False), train_iters=2, reduced=True)
    output = torch.log_softmax(torch.randn(3, 2), 1)
    target = torch.rand(3, 2)
    torch.testing.assert_close(model.loss(output, target), F.mse_loss(output.exp(), target))


def test_unknown_soft_target_loss_is_rejected(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.eval_loss = "CE"
    with pytest.raises(NotImplementedError):
        GCN(8, 8, 2, args).fit_with_val(soft_graph(), train_iters=1, reduced=True)


def test_single_output_class_uses_binary_cross_entropy(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.multi_label = True
    args.metric = lambda output, labels: float(((output > 0.5).float() == labels).float().mean())
    graph = tiny_graph()
    graph.nclass = 1
    graph.labels_train = graph.labels_train.float()[:, None]
    graph.labels_val = graph.labels_val.float()[:, None]
    model = GCN(8, 8, 1, args)
    score = model.fit_with_val(graph, train_iters=3)
    assert isinstance(model.loss, torch.nn.BCELoss)
    assert 0 <= score <= 1


def test_attack_mode_preset_loss_and_unreachable_best_value_keep_last_weights(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    model = GCN(8, 8, 2, args, mode="attack")
    assert model.loss is F.nll_loss
    score = model.fit_with_val(tiny_graph(), train_iters=2, best_val=2.0, verbose=True)
    assert score == 2.0


def test_final_output_returns_none_and_invalid_validation_interval_fails(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    model = GCN(8, 8, 2, args)
    assert model.fit_with_val(tiny_graph(), train_iters=2, final_output=True) is None
    args.val_interval = 0
    with pytest.raises(ValueError, match="val_interval"):
        model.fit_with_val(tiny_graph(), train_iters=1)


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_verbose_test_reports_metric_on_held_out_nodes(tmp_path, capsys, setting):
    args = quick_args(tmp_path, "random", setting)
    graph = tiny_graph()
    model = GCN(8, 8, 2, args)
    model.fit_with_val(graph, train_iters=3, setting=setting)
    accuracy = model.test(graph, setting=setting, verbose=True)
    assert "Test set results" in capsys.readouterr().out
    features = graph.feat_test if setting == "ind" else graph.feat_full
    adjacency = graph.adj_test if setting == "ind" else graph.adj_full
    predictions = model.predict(features, adjacency).argmax(1)
    if setting == "trans":
        predictions = predictions[graph.idx_test]
    assert accuracy == pytest.approx((predictions == graph.labels_test).float().mean().item())


def test_graphsage_msgc_keeps_raw_adjacency_and_simgc_uses_sparse_normalization(
    tmp_path, monkeypatch
):
    import graphslim.models.base as base

    calls = []
    original = base.normalize_adj_tensor

    def record(adjacency, sparse=False):
        calls.append(sparse)
        return original(adjacency, sparse=sparse)

    monkeypatch.setattr(base, "normalize_adj_tensor", record)
    graph = soft_graph()
    graph.labels_syn = graph.labels_train
    args = quick_args(tmp_path, "msgc", "trans")
    GraphSage(8, 8, 2, args).fit_with_val(graph, train_iters=1, reduced=True)
    # Only the validation graph is normalized for msgc GraphSage.
    assert calls == [True]
    calls.clear()
    args = quick_args(tmp_path, "simgc", "trans")
    GCN(8, 8, 2, args).fit_with_val(graph, train_iters=1, reduced=True)
    assert calls == [True, True]


def test_gat_fit_accepts_batched_dense_synthetic_adjacency(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    graph = tiny_graph()
    graph.feat_syn = graph.feat_train[:8]
    graph.adj_syn = torch.stack([torch.eye(8), ring(8)])
    graph.labels_syn = torch.cat([graph.labels_train[:8]] * 2)
    model = GAT(8, 2, 2, args, mode="eval")
    score = model.fit_with_val(graph, train_iters=2, reduced=True)
    assert 0 <= score <= 1
    # A single dense graph is converted to a normalized sparse adjacency.
    assert 0 <= model.fit_with_val(graph, train_iters=2) <= 1
    assert 0 <= model.test(graph) <= 1


@pytest.mark.parametrize("name", ["GCN", "Cheby", "GraphSage", "APPNP", "SGC"])
def test_deep_models_without_batch_norm_agree_on_dense_and_sparse_graphs(tmp_path, name):
    args = quick_args(tmp_path, "random", "trans")
    args.nlayers, args.ntrans, args.with_bn, args.activation = 3, 3, False, "none"
    torch.manual_seed(9)
    model = get_model_class(name)(4, 5, 2, args).eval()
    assert len(model.layers) == 3 and not hasattr(model, "bns")
    x, adjacency = torch.randn(7, 4), normalize_adj_tensor(ring(7))
    dense = model(x, adjacency)
    assert dense.shape == (7, 2)
    torch.testing.assert_close(model(x, SparseTensor.from_dense(adjacency)), dense)


def test_attention_layer_averages_heads_and_respects_explicit_size():
    torch.manual_seed(10)
    layer = GATConv(3, 2, heads=3, concat=False, bias=True)
    assert layer.bias.shape == (2,)
    x = torch.randn(5, 3)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    assert layer(x, edges).shape == (5, 2)
    # With size=(5, 3) only the first three targets receive self loops.
    bipartite = GATConv((3, 3), 2)
    output, (index, _) = bipartite(
        (x, x[:3]), edges[:, :2], size=(5, 3), return_attention_weights=True
    )
    assert output.shape == (3, 2)
    assert {(i, i) for i in range(3)} <= set(zip(*index.tolist()))
    no_destination = GATConv((3, 3), 2)
    assert no_destination((x, None), edges, size=(5, 5)).shape == (5, 2)


def test_gat_fit_converts_dense_synthetic_adjacency_to_sparse(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    graph = tiny_graph()
    graph.feat_syn, graph.adj_syn = graph.feat_train[:8], ring(8)
    graph.labels_syn = graph.labels_train[:8]
    model = GAT(8, 2, 2, args, mode="cross")
    assert 0 <= model.fit_with_val(graph, train_iters=2, reduced=True) <= 1


def test_gat_default_training_mode_builds_two_attention_layers(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.nlayers, args.dropout = 3, 0.0
    model = GAT(4, 3, 2, args).eval()
    assert (model.heads, model.output_heads, model.nlayers) == (8, 1, 3)
    assert model.conv1.heads == 8 and model.conv2.in_channels == 24
    edges = SparseTensor.from_dense(ring())
    output = model(torch.randn(6, 4), edges)
    assert output.shape == (6, 2)
    torch.testing.assert_close(output.exp().sum(1), torch.ones(6))

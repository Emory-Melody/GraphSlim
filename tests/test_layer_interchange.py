"""Check layer outputs and gradients against dense algebra and PyG attention."""

from copy import deepcopy

import pytest
import torch
from torch_geometric.nn import GATConv as PyGGATConv
from torch_sparse import SparseTensor

from graphslim.models.appnp import APPNP, SparseDropout
from graphslim.models.layers import (
    ChebConvolution,
    GATConv,
    GraphConvolution,
    MyLinear,
    SageConvolution,
)
from test_lightweight_methods import quick_args


@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("bipartite", [False, True])
def test_attention_matches_pyg_outputs_weights_and_input_gradients(concat, sparse, bipartite):
    torch.manual_seed(4)
    channels = (3, 5) if bipartite else 3
    layer = GATConv(channels, 2, heads=2, concat=concat, bias=concat)
    reference = PyGGATConv(channels, 2, heads=2, concat=concat, bias=concat, residual=False)
    with torch.no_grad():
        if bipartite:
            reference.lin_src.weight.copy_(layer.lin_l.weight)
            reference.lin_dst.weight.copy_(layer.lin_r.weight)
        else:
            reference.lin.weight.copy_(layer.lin_l.weight)
        reference.att_src.copy_(layer.att_l)
        reference.att_dst.copy_(layer.att_r)
        if concat:
            reference.bias.copy_(layer.bias)
    x = torch.randn(6, 3, requires_grad=True)
    y = torch.randn(4, 5, requires_grad=True) if bipartite else None
    x_ref = x.detach().clone().requires_grad_()
    y_ref = y.detach().clone().requires_grad_() if y is not None else None
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 0, 1, 2, 0]])
    adjacency = (
        SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 4 if bipartite else 6)).t()
        if sparse
        else edge_index
    )
    inputs, ref_inputs = ((x, y), (x_ref, y_ref)) if bipartite else (x, x_ref)
    output, attention = layer(inputs, adjacency, return_attention_weights=True)
    expected, expected_attention = reference(ref_inputs, adjacency, return_attention_weights=True)
    torch.testing.assert_close(output, expected)
    if sparse:
        torch.testing.assert_close(attention.to_dense(), expected_attention.to_dense())
    else:
        assert torch.equal(attention[0], expected_attention[0])
        torch.testing.assert_close(attention[1], expected_attention[1])
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    if bipartite:
        torch.testing.assert_close(y.grad, y_ref.grad)
    assert "heads=2" in repr(layer)


def test_attention_without_destination_features_matches_pyg():
    layer = GATConv((3, 3), 2, add_self_loops=False, bias=False)
    x = torch.ones(4, 3)
    edges = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    actual, (_, weights) = layer((x, None), edges, size=(4, 2), return_attention_weights=True)
    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, layer.lin_l(x)[:2])
    torch.testing.assert_close(weights, torch.full((4, 1), 0.5))


@pytest.mark.parametrize("kind", ["graph", "sage", "cheb"])
@pytest.mark.parametrize("bias", [False, True])
def test_sparse_and_dense_layer_outputs_and_gradients_agree(kind, bias):
    torch.manual_seed(7)
    adjacency = torch.tensor([[0.0, 0.5, 0], [0.5, 0, 0.8], [0, 0.8, 0]])
    layers = {
        "graph": lambda: GraphConvolution(3, 2, with_bias=bias, weight_initializer="glorot"),
        "sage": lambda: SageConvolution(3, 2),
        "cheb": lambda: ChebConvolution(3, 2, with_bias=bias, single_param=bias, K=4),
    }
    dense = layers[kind]()
    sparse = deepcopy(dense)
    x = torch.randn(3, 3, requires_grad=True)
    x_sparse = x.detach().clone().requires_grad_()
    output = dense(x, adjacency).reshape(3, 2)
    expected = sparse(x_sparse, SparseTensor.from_dense(adjacency))
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_sparse.grad)
    assert "3" in repr(dense)


def test_chebyshev_polynomial_agrees_with_scalar_eigendecomposition():
    layer = ChebConvolution(1, 1, with_bias=False, single_param=False, K=4).double()
    with torch.no_grad():
        for value, lin in enumerate(layer.lins, 1):
            lin.weight.fill_(value)
    eigenvalues = torch.tensor([-0.5, 0, 0.5], dtype=torch.float64)
    adjacency = torch.diag(eigenvalues)
    actual = layer(torch.ones(3, 1, dtype=torch.float64), adjacency).flatten()
    expected = (
        1
        + 2 * eigenvalues
        + 3 * (2 * eigenvalues.square() - 1)
        + 4 * (4 * eigenvalues.pow(3) - 3 * eigenvalues)
    )
    torch.testing.assert_close(actual, expected)


def test_linear_layer_has_trainable_bias_and_rejects_wrong_feature_width():
    layer = MyLinear(3, 2)
    torch.testing.assert_close(layer(torch.zeros(4, 3)), layer.bias.expand(4, 2))
    with pytest.raises(RuntimeError):
        layer(torch.ones(4, 4))
    assert "3 -> 2" in repr(layer)


@pytest.mark.parametrize("transforms", [1, 3, 4])
@pytest.mark.parametrize("multilabel", [False, True])
def test_appnp_neighbor_blocks_equal_full_graph_propagation(tmp_path, transforms, multilabel):
    args = quick_args(tmp_path, "random", "trans")
    args.ntrans, args.with_bn, args.multi_label = transforms, True, multilabel
    args.dropout, args.activation = 0, "relu"
    model = APPNP(3, 4, 2, args).eval()
    x = torch.arange(12).reshape(4, 3).float()
    adjacency = torch.eye(4).roll(1, 0)
    block = (SparseTensor.from_dense(adjacency), None, (4, 4))
    full = model(x, adjacency)
    sampled = model.forward_sampler(x, [block] * model.nlayers)
    if multilabel:
        # The full path returns normalized log scores. Compare class ordering,
        # while the sampled multilabel path has independent sigmoid outputs.
        assert torch.equal(full.argmax(1), sampled.argmax(1))
        assert sampled.min() >= 0 and sampled.max() <= 1
    else:
        torch.testing.assert_close(sampled, full)


def test_sparse_dropout_scales_kept_edges_and_preserves_evaluation_input():
    values = torch.arange(1, 101).float()
    edges = torch.arange(100).repeat(2, 1)
    adjacency = torch.sparse_coo_tensor(edges, values, (100, 100)).coalesce()
    dropout = SparseDropout(0.5)
    assert dropout(adjacency, training=False) is adjacency
    torch.manual_seed(12)
    result = dropout(adjacency, training=True).coalesce()
    torch.testing.assert_close(result.values(), 2 * values[result.indices()[0]])
    assert 20 < result._nnz() < 80

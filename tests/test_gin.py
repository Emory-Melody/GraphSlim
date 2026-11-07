# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""GIN parity with PyG and real node-evaluation integration."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.nn import DenseGINConv, GINConv
from torch_sparse import SparseTensor

from graphslim.config import get_args
from graphslim.evaluation import Evaluator
from graphslim.models import DOWNSTREAM_MODELS, get_model_class
from graphslim.models.gin import GIN, GINConvolution
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("layout", ["dense", "coo", "csr", "torch_sparse"])
@pytest.mark.parametrize("train_eps", [False, True])
def test_unweighted_outputs_and_gradients_match_pyg(layout, train_eps):
    torch.manual_seed(12)
    layer = GINConvolution(3, 2, hidden_features=5, eps=0.2, train_eps=train_eps).double()
    reference = GINConv(deepcopy(layer.nn), eps=0.2, train_eps=train_eps).double()
    reference.load_state_dict(layer.state_dict())
    # Asymmetric edges and an isolated node detect adjacency transposition errors.
    edges = torch.tensor([[0, 0, 2, 3], [1, 2, 1, 0]])
    adjacency = torch.zeros(5, 5, dtype=torch.float64)
    adjacency[edges[1], edges[0]] = 1
    inputs = {
        "dense": lambda: adjacency,
        "coo": lambda: adjacency.to_sparse(),
        "csr": lambda: adjacency.to_sparse_csr(),
        "torch_sparse": lambda: SparseTensor.from_dense(adjacency),
    }
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    actual, expected = layer(x, inputs[layout]()), reference(x_ref, edges)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize(
    "layout,batch",
    [
        ("dense", False),
        ("coo", False),
        ("csr", False),
        ("torch_sparse", False),
        ("dense", True),
    ],
)
def test_weighted_outputs_and_adjacency_gradients_match_pyg_dense(layout, batch):
    torch.manual_seed(14)
    layer = GINConvolution(3, 2, hidden_features=4, train_eps=True).double()
    reference = DenseGINConv(deepcopy(layer.nn), train_eps=True).double()
    reference.load_state_dict(layer.state_dict())
    shape = (2, 4, 4) if batch else (4, 4)
    feature_shape = (2, 4, 3) if batch else (4, 3)
    adj = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
    adj_ref = adj.detach().clone().requires_grad_()
    x = torch.randn(*feature_shape, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    adjacency = {
        "dense": lambda: adj,
        "coo": lambda: adj.to_sparse(),
        "csr": lambda: adj.to_sparse_csr(),
        "torch_sparse": lambda: SparseTensor.from_dense(adj),
    }[layout]()
    actual, expected = layer(x, adjacency), reference(x_ref, adj_ref)
    if not batch:
        expected = expected.squeeze(0)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(adj.grad, adj_ref.grad)
    torch.testing.assert_close(layer.eps.grad, reference.eps.grad)


@pytest.mark.parametrize("layers", [1, 3])
def test_batched_synthetic_features_and_batchnorm(tmp_path, layers):
    args = quick_args(tmp_path, "msgc", "trans")
    args.dropout, args.with_bn, args.nlayers = 0, True, layers
    model = GIN(3, 4, 2, args).eval()
    model.initialize()
    x, adj = torch.randn(2, 4, 3), torch.rand(2, 4, 4)
    expected = torch.cat([model(x[i], adj[i]) for i in range(2)])
    actual = model(x.flatten(0, 1), adj)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("reduced", [False, True])
def test_training_cache_parity_raw_sum_and_checkpoint_reload(tmp_path, setting, reduced):
    args = quick_args(tmp_path, "random", setting)
    args.eval_epochs, args.gin_train_eps = 4, True
    graph = tiny_graph()
    if reduced:
        graph = create_reducer("random", setting, graph, args).reduce(graph)
    original_adjacency = graph.adj_full.copy()
    states, scores, predictions = [], [], []
    for cached in (False, True):
        seed_everything(17)
        args.eval_cache = cached
        evaluator = Evaluator(args)
        seen_adjacencies = []
        # Check the actual shared training/validation path, not only direct forward().
        model_class = get_model_class("GIN")
        original_forward = model_class.forward

        def observe(self, features, adjacency, **kwargs):
            dense = adjacency.to_dense() if hasattr(adjacency, "to_dense") else adjacency
            seen_adjacencies.append(dense.detach().clone())
            return original_forward(self, features, adjacency, **kwargs)

        with patch.object(model_class, "forward", observe):
            score = evaluator.test(graph, "GIN", reduced=reduced, mode="cross", verbose=False)
        model = evaluator.last_model
        assert np.isfinite(score).all() and all(0 <= value <= 1 for value in score)
        source = (
            graph.adj_syn
            if reduced
            else (graph.adj_full if setting == "trans" else graph.adj_train)
        )
        expected = torch.as_tensor(source.toarray()) if hasattr(source, "toarray") else source
        if hasattr(expected, "to_dense"):
            expected = expected.to_dense()
        torch.testing.assert_close(seen_adjacencies[0], expected.float())
        heldout = graph.adj_full if setting == "trans" else graph.adj_val
        torch.testing.assert_close(seen_adjacencies[1], torch.as_tensor(heldout.toarray()).float())
        prediction = model.predict(graph.feat_full, graph.adj_full)
        direct = model(graph.feat_full, torch.as_tensor(graph.adj_full.toarray()).float())
        torch.testing.assert_close(prediction, direct)
        states.append(deepcopy(model.state_dict()))
        scores.append(score)
        predictions.append(prediction)
    assert scores[0] == scores[1]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    assert (graph.adj_full != original_adjacency).nnz == 0
    checkpoint = tmp_path / "gin.pt"
    torch.save(states[1], checkpoint)
    restored = GIN(8, args.hidden, 2, args, mode="cross")
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.predict(graph.feat_full, graph.adj_full), predictions[1])


def test_cli_selects_gin_and_epsilon_options(tmp_path):
    args = get_args(
        [
            "-G",
            "-1",
            "--eval_model",
            "GIN",
            "--gin_eps",
            "0.3",
            "--gin_train_eps",
            "--save_path",
            str(tmp_path),
        ]
    )
    assert args.eval_model == "GIN" and "GIN" in DOWNSTREAM_MODELS
    model = get_model_class(args.eval_model)(8, 4, 2, args)
    for layer in model.layers:
        assert isinstance(layer.eps, torch.nn.Parameter)
        torch.testing.assert_close(layer.eps, torch.tensor([0.3]))
    with torch.no_grad():
        model.layers[0].eps.add_(3)
    model.initialize()
    torch.testing.assert_close(model.layers[0].eps, torch.tensor([0.3]))

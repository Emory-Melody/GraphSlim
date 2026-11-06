"""GCGP kernels, Gaussian-process prediction and reducer option handling."""

import numpy as np
import pytest
import torch

from graphslim.condensation import gcgp
from graphslim.dataset import load_reduced
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_released_kernel_matches_package_kernel_and_cached_diagonal():
    torch.manual_seed(0)
    x, y = torch.rand(5, 3, dtype=torch.float64), torch.rand(4, 3, dtype=torch.float64)
    released = gcgp.released_neural_kernel(x, y, layers=2)
    torch.testing.assert_close(released, gcgp.neural_kernel(x, y, layers=2), rtol=1e-5, atol=1e-6)
    diagonal = (x * x).sum(1, keepdim=True)
    torch.testing.assert_close(gcgp.released_neural_kernel(x, y, 2, diagonal_x=diagonal), released)
    gram = gcgp.released_neural_kernel(x, x, layers=1)
    torch.testing.assert_close(gram, gram.T)


def test_tangent_kernel_with_structure_is_symmetric_and_identity_neutral():
    torch.manual_seed(1)
    x = torch.rand(4, 3, dtype=torch.float64)
    plain = gcgp.tangent_kernel(x, x, layers=2)
    torch.testing.assert_close(plain, plain.T)
    identity = torch.eye(4, dtype=torch.float64)
    torch.testing.assert_close(gcgp.tangent_kernel(x, x, identity, identity, 2, hops=1), plain)
    ring = torch.roll(identity, 1, 0) + identity
    propagated = gcgp.tangent_kernel(x, x, ring, ring, 2, hops=2)
    torch.testing.assert_close(propagated, propagated.T)
    assert not torch.allclose(propagated, plain)


def test_concrete_adjacency_sampling_and_thresholding():
    alpha = torch.tensor([[0.0, 3.0], [1.0, 0.0]])
    evaluated = gcgp.concrete_adjacency(alpha, training=False)
    torch.testing.assert_close(evaluated, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    uniform = torch.full((2, 2), 0.5)
    sampled = gcgp.concrete_adjacency(alpha, epoch=0, uniform=uniform)
    # alpha=2 after symmetrization: sigmoid(log 2 / 0.3) on the off-diagonal.
    expected = torch.sigmoid(torch.log(torch.tensor(2.0)) / 0.3)
    torch.testing.assert_close(sampled[0, 1], expected)
    torch.manual_seed(0)
    random = gcgp.concrete_adjacency(alpha, epoch=500)
    torch.testing.assert_close(random, random.T)
    assert random.diagonal().eq(1).all()


def test_gp_prediction_paths_and_ridge_validation():
    kernel = torch.tensor([[2.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    targets = torch.eye(2, dtype=torch.float64)
    package = gcgp.gp_predict(kernel, kernel, targets, 0.1)
    released = gcgp.gp_predict(kernel, kernel, targets, 0.1, released_arithmetic=True)
    torch.testing.assert_close(package, released)
    torch.testing.assert_close(package.sum(1), torch.ones(2, dtype=torch.float64))
    assert package[0, 0] > package[0, 1]
    with pytest.raises(ValueError, match="positive"):
        gcgp.gp_predict(kernel, kernel, targets, 0)


def gcgp_args(tmp_path, **options):
    args = quick_args(tmp_path, "gcgp", "trans")
    args.epochs = 3
    for key, value in options.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize(
    "options",
    [
        {"gcgp_kernel": "NTK", "gcgp_batch_size": 8},
        {"gcgp_kernel": "SNTK", "gcgp_learn_adj": True, "gcgp_batch_size": 8},
        {"gcgp_arithmetic": "released", "record_loss": True},
    ],
)
def test_gcgp_variants_reduce_to_finite_soft_label_graphs(tmp_path, options):
    args = gcgp_args(tmp_path, **options)
    graph = tiny_graph()
    seed_everything(1)
    reduced = create_reducer("gcgp", "trans", graph, args).reduce(graph, verbose=False)
    assert reduced.feat_syn.shape == (8, 8)
    assert reduced.adj_syn.shape == (8, 8)
    adjacency = reduced.adj_syn.to_dense() if reduced.adj_syn.is_sparse else reduced.adj_syn
    torch.testing.assert_close(adjacency, adjacency.T)
    assert torch.isfinite(reduced.feat_syn).all()
    _, features, labels = load_reduced(args)
    assert features.shape == (8, 8) and len(labels) == 8
    if options.get("record_loss"):
        trace = args._reduction_loss_trace
        assert len(trace) == 3 and np.isfinite(trace).all()


@pytest.mark.parametrize(
    "options,message",
    [
        ({"gcgp_kernel": "RBF"}, "gcgp_kernel"),
        ({"gcgp_learn_adj": True}, "requires SNTK"),
        ({"gcgp_batch_size": 0}, "batch_size"),
        ({"gcgp_arithmetic": "fast"}, "arithmetic"),
    ],
)
def test_gcgp_rejects_invalid_options(tmp_path, options, message):
    args = gcgp_args(tmp_path, **options)
    graph = tiny_graph()
    with pytest.raises(ValueError, match=message):
        create_reducer("gcgp", "trans", graph, args).reduce(graph, verbose=False)


def test_gcgp_stops_on_non_finite_loss(tmp_path, monkeypatch):
    args = gcgp_args(tmp_path)
    graph = tiny_graph()
    monkeypatch.setattr(
        gcgp, "gp_predict", lambda ts, ss, targets, *rest: ts @ targets * float("nan")
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        create_reducer("gcgp", "trans", graph, args).reduce(graph, verbose=False)

"""Numerical checks for the GCSNTK kernel, kernel ridge regression, and SGDD spectral maps.

The legacy version of this file aborted the interpreter: ``sparse_kron`` used to scale
the index buffer of an already coalesced input in place, so passing the same
coalesced tensor as both operands produced out-of-range COO indices and a native
crash in ``to_dense``. The regression tests below use coalesced inputs on purpose.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from graphslim.models.ignr import GraphonLearner, mx_inv, mx_inv_sqrt
from graphslim.models.krr import KernelRidgeRegression
from graphslim.models.sntk import StructureBasedNeuralTangentKernel


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def binary_sparse(dense):
    index = dense.nonzero().T
    return torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), tuple(dense.shape))


@pytest.mark.parametrize("same_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_kronecker_never_mutates_coalesced_inputs(same_input, dtype):
    first_dense = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=dtype)
    second_dense = first_dense if same_input else torch.eye(2, dtype=dtype)
    first = first_dense.to_sparse()
    second = first if same_input else second_dense.to_sparse()
    assert first.is_coalesced() and second.is_coalesced()
    saved = [(value.indices().clone(), value.values().clone()) for value in [first, second]]
    kernel = StructureBasedNeuralTangentKernel()
    for _ in range(3):
        actual = kernel.sparse_kron(first, second)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual.to_dense(), torch.kron(first_dense, second_dense))
        for value, (indices, weights) in zip([first, second], saved):
            assert torch.equal(value.indices(), indices)
            assert torch.equal(value.values(), weights)


def test_sparse_kronecker_uses_only_the_sparsity_pattern():
    weighted = torch.tensor([[0.0, 2.0], [3.0, 0.0]]).to_sparse()
    actual = StructureBasedNeuralTangentKernel().sparse_kron(weighted, torch.eye(1).to_sparse())
    torch.testing.assert_close(actual.to_dense(), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


@pytest.mark.parametrize("same_input", [False, True])
def test_sparse_kronecker_matches_dense_binary_product(same_input):
    first_dense = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    second_dense = first_dense if same_input else torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    first = binary_sparse(first_dense)
    second = first if same_input else binary_sparse(second_dense)
    kernel = StructureBasedNeuralTangentKernel()
    for _ in range(3):
        actual = kernel.sparse_kron(first, second)
        assert actual.shape == (
            first.shape[0] * second.shape[0],
            first.shape[1] * second.shape[1],
        )
        torch.testing.assert_close(actual.to_dense(), torch.kron(first_dense, second_dense))
    torch.testing.assert_close(first.to_dense(), first_dense)


@pytest.mark.parametrize("scale", ["add", "average"])
@pytest.mark.parametrize("zero_features", [False, True])
def test_sntk_gram_stays_finite_symmetric_psd_and_repeatable(scale, zero_features):
    torch.manual_seed(3)
    features = torch.rand(9, 5)
    if zero_features:
        features[:2] = 0  # regression: zero-variance rows used to give 0/0 = NaN
    features.requires_grad_()
    ring = torch.eye(9) + torch.eye(9).roll(1, 0) + torch.eye(9).roll(-1, 0)
    adjacency = torch.eye(9).to_sparse() if zero_features else ring.to_sparse()
    kernel = StructureBasedNeuralTangentKernel(K=2, L=2, scale=scale)
    # The same coalesced operator is passed twice, as in K_ss = kernel(G_s, G_s, E_s, E_s).
    first = kernel.nodes_gram(features, features, adjacency, adjacency)
    second = kernel.nodes_gram(features, features, adjacency, adjacency)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    torch.testing.assert_close(first, first.T)
    assert torch.linalg.eigvalsh(first.double()).min() > -1e-4 * first.diag().max()
    first.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    if zero_features:
        # Zero-variance nodes have (numerically) zero covariance with every node.
        assert first[:2].abs().max() < 1e-12 * first.abs().max()
    assert torch.equal(adjacency.to_dense(), torch.eye(9) if zero_features else ring)


def test_sntk_positive_variance_path_is_unchanged_by_zero_variance_guard():
    torch.manual_seed(4)
    features = torch.rand(6, 3)
    sigma = features @ features.T
    updated, diag = StructureBasedNeuralTangentKernel().update_diag(sigma)
    assert torch.equal(diag, torch.sqrt(torch.diag(sigma)))
    assert torch.isfinite(updated).all()


def test_sntk_identity_structure_reduces_to_relu_ntk_of_features():
    # With an identity graph and K=1, L=1 the kernel is the two-layer ReLU NTK.
    torch.manual_seed(5)
    features = torch.rand(4, 3)
    identity = binary_sparse(torch.eye(4))
    gram = StructureBasedNeuralTangentKernel(K=1, L=1).nodes_gram(
        features, features, identity, binary_sparse(torch.eye(4))
    )
    sigma = features @ features.T
    std = sigma.diag().sqrt()
    c = (sigma / std[:, None] / std[None, :]).clamp(-0.9999, 0.9999)
    relu = (c * (torch.pi - c.acos()) + (1 - c * c).sqrt()) / torch.pi
    expected = sigma * (torch.pi - relu.acos()) / torch.pi + relu * std[:, None] * std[None, :]
    torch.testing.assert_close(gram, expected)


@pytest.mark.parametrize("ridge", [0.0, 1e-6])
def test_full_rank_krr_has_finite_predictions_and_gradients(ridge):
    torch.manual_seed(8)
    support = torch.rand(8, 10, requires_grad=True)
    targets = torch.rand(8, 3, requires_grad=True)
    query = torch.rand(7, 10)
    labels = torch.nn.functional.one_hot(torch.arange(7) % 3, 3)
    model = KernelRidgeRegression(lambda a, b, *_: a @ b.T, ridge)
    prediction, accuracy = model(query, support, labels, targets, None, None)
    assert prediction.shape == (7, 3)
    torch.testing.assert_close(prediction.sum(1), torch.ones(7))
    expected = (prediction.argmax(1) == labels.argmax(1)).float().mean().item()
    assert accuracy == pytest.approx(expected)
    prediction.square().mean().backward()
    for value in [prediction, support.grad, targets.grad]:
        assert torch.isfinite(value).all()


def test_krr_singular_system_raises_linalg_error():
    support = torch.zeros(4, 3, requires_grad=True)
    model = KernelRidgeRegression(lambda a, b, *_: a @ b.T, 0.0)
    labels = torch.eye(2)[torch.tensor([0, 1, 0])]
    with pytest.raises(torch.linalg.LinAlgError):
        model(torch.rand(3, 3), support, labels, torch.rand(4, 2), None, None)


def test_krr_matches_original_well_conditioned_objective_and_gradient():
    torch.manual_seed(19)
    support = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    query = torch.randn(5, 7, dtype=torch.float64)
    labels = torch.eye(3)[torch.arange(5) % 3]
    ridge = 1e-4
    model = KernelRidgeRegression(lambda a, b, *_: a @ b.T, ridge)
    actual, _ = model(query, support, labels, targets, None, None)
    gram = support @ support.T
    coefficients = torch.linalg.solve(gram + ridge * gram.trace() / 4 * torch.eye(4), targets)
    expected = (query @ support.T @ coefficients).softmax(1)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), [support, targets], retain_graph=True),
        torch.autograd.grad(expected.square().sum(), [support, targets]),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("indefinite", [False, True])
def test_sgdd_spectral_maps_match_svd_away_from_cutoff(indefinite):
    torch.manual_seed(6)
    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    values = torch.tensor([0.4, 0.8, 1.3, 2.1], dtype=torch.float64)
    if indefinite:
        values[2] *= -1
    matrix = rotation @ values.diag() @ rotation.T
    u, singular, vh = torch.linalg.svd(matrix)
    root, inverse = mx_inv_sqrt(matrix)
    torch.testing.assert_close(root, (u / singular.sqrt()) @ vh, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(inverse, (u / singular) @ vh, atol=1e-12, rtol=1e-12)


def test_sgdd_indefinite_spectral_map_has_correct_gradients():
    matrix = torch.tensor(
        [[2.0, 0.3, 0.1], [0.3, -1.0, 0.2], [0.1, 0.2, 0.8]], dtype=torch.float64
    ).requires_grad_()
    assert torch.autograd.gradcheck(mx_inv_sqrt, (matrix,), atol=1e-5, rtol=1e-4)


def test_sgdd_repeated_singular_values_survive_actual_graphon_training():
    torch.manual_seed(21)
    args = SimpleNamespace(ep_ratio=0.5, sinkhorn_iter=10, mx_size=12)
    learner = GraphonLearner(4, nfeat=8, nnodes=8, args=args, device="cpu")
    original = torch.ones(12, 12) - torch.eye(12)
    learner.Lx_inv = mx_inv(original, "cpu")
    features = torch.rand(8, 4, requires_grad=True)
    optimizer = torch.optim.Adam([features, *learner.parameters()], lr=0.01)
    for _ in range(40):
        optimizer.zero_grad()
        adjacency, loss = learner(features)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(features.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in learner.parameters() if p.grad is not None)
        optimizer.step()
    assert torch.isfinite(adjacency).all()


@pytest.mark.parametrize("method", ["gcsntk", "sgdd"])
def test_reducers_save_graphs_and_complete_gcn_training(tmp_path, method):
    from graphslim.dataset import load_reduced
    from graphslim.dataset.loader import TransAndInd
    from graphslim.evaluation import Evaluator
    from graphslim.reduction import create_reducer
    from graphslim.utils import seed_everything
    from test_lightweight_methods import quick_args, tiny_graph

    args = quick_args(tmp_path, method, "trans")
    args.dataset = "cora"
    args.epochs, args.eval_epochs, args.run_inter_eval = 4, 20, 1
    args.checkpoints = [3]
    args.outer_loop, args.inner_loop, args.mx_size = 1, 1, 16
    args.condense_model = "SGC"
    args.K, args.L, args.ridge, args.scale, args.adj = 1, 1, 1e-6, "add", False
    graph = tiny_graph()
    graph.class_dict = graph.class_dict2 = graph.samplers = None
    graph.retrieve_class = MethodType(TransAndInd.retrieve_class, graph)
    graph.retrieve_class_sampler = MethodType(TransAndInd.retrieve_class_sampler, graph)
    seed_everything(1)
    reduced = create_reducer(method, "trans", graph, args).reduce(graph, verbose=False)
    adjacency, features, labels = load_reduced(args)
    assert torch.isfinite(adjacency).all() and torch.isfinite(features).all()
    assert len(features) == len(labels) == adjacency.shape[0]
    reduced.adj_syn, reduced.feat_syn, reduced.labels_syn = adjacency, features, labels
    validation, test = Evaluator(args).test(
        reduced, "GCN", reduced=True, mode="cross", verbose=False
    )
    assert 0 <= validation <= 1 and 0 <= test <= 1

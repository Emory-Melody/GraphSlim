"""Compare the structure-based NTK block kernel to a full joint covariance recursion."""

import math

import pytest
import torch

from graphslim.models.sntk import StructureBasedNeuralTangentKernel


def binary_sparse(dense):
    # ``sparse_kron`` treats the operators as binary and builds float32 ones.
    # Fresh (uncoalesced) COO tensors also keep the inputs' index buffers private.
    index = dense.nonzero().T
    return torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), tuple(dense.shape))


def joint_recursion(left, right, a, b, aggregations, layers, scale):
    graph = torch.block_diag(a, b)
    if scale == "average":
        graph = graph / graph.sum(1, keepdim=True)
    features = torch.cat((left, right))
    sigma = features @ features.T
    theta = sigma
    for _ in range(aggregations):
        sigma = graph @ sigma @ graph.T
        theta = graph @ theta @ graph.T
        for _ in range(layers):
            std = sigma.diag().clamp_min(1e-12).sqrt()
            correlation = (sigma / (std[:, None] * std[None, :])).clamp(-0.9999, 0.9999)
            covariance = (
                correlation * (math.pi - correlation.acos()) + (1 - correlation.square()).sqrt()
            ) / math.pi
            sigma = covariance * std[:, None] * std[None, :]
            theta = theta * (math.pi - covariance.acos()) / math.pi + sigma
    return theta[: len(left), len(left) :]


# For K > 1 and L > 1 the per-graph variance recursion in ``diag`` applies one
# ReLU update per stage instead of L, so only these configurations are exact.
@pytest.mark.parametrize(
    "aggregations,layers,rtol", [(1, 1, 1e-5), (2, 1, 1e-5), (1, 2, 1e-3), (1, 3, 1e-3)]
)
@pytest.mark.parametrize("scale", ["add", "average"])
def test_cross_kernel_and_gradients_match_joint_recursion(aggregations, layers, rtol, scale):
    torch.manual_seed(41)
    left = torch.randn(3, 4, requires_grad=True)
    right = torch.randn(2, 4, requires_grad=True)
    a = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    b = torch.ones(2, 2)
    expected = joint_recursion(
        left.double(), right.double(), a.double(), b.double(), aggregations, layers, scale
    )
    model = StructureBasedNeuralTangentKernel(K=aggregations, L=layers, scale=scale)
    actual = model.nodes_gram(left, right, binary_sparse(a), binary_sparse(b))
    assert actual.shape == (3, 2) and actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=1e-6)
    observed_grad = torch.autograd.grad(actual.sum(), (left, right))
    expected_grad = torch.autograd.grad(expected.sum(), (left, right))
    for observed, reference in zip(observed_grad, expected_grad):
        torch.testing.assert_close(observed, reference, rtol=10 * rtol, atol=1e-5)

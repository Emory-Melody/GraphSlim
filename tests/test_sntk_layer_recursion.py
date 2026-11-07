# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Compare the structure-based NTK block kernel to a full joint covariance recursion."""

import math

import pytest
import torch

from graphslim.models.sntk import StructureBasedNeuralTangentKernel


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


# ``diag`` refreshes the per-graph variance once per aggregation stage, as in the
# released GC-SNTK code, so for K > 1 and L > 1 the kernel is not the exact joint
# recursion; only these configurations are exact.
@pytest.mark.parametrize(
    "aggregations,layers,rtol", [(1, 1, 1e-9), (2, 1, 1e-9), (1, 2, 1e-3), (1, 3, 1e-3)]
)
@pytest.mark.parametrize("scale", ["add", "average"])
def test_cross_kernel_and_gradients_match_joint_recursion(aggregations, layers, rtol, scale):
    torch.manual_seed(41)
    left = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    b = torch.ones(2, 2, dtype=torch.float64)
    expected = joint_recursion(left, right, a, b, aggregations, layers, scale)
    model = StructureBasedNeuralTangentKernel(K=aggregations, L=layers, scale=scale)
    # Coalesced float64 operators: regression for in-place index scaling and dtype.
    actual = model.nodes_gram(left, right, a.to_sparse(), b.to_sparse())
    assert actual.shape == (3, 2) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=1e-9)
    observed_grad = torch.autograd.grad(actual.sum(), (left, right), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (left, right))
    for observed, reference in zip(observed_grad, expected_grad):
        torch.testing.assert_close(observed, reference, rtol=10 * rtol, atol=1e-8)


def test_variance_is_refreshed_once_per_aggregation_stage():
    torch.manual_seed(2)
    features = torch.randn(3, 4, dtype=torch.float64)
    graph = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    kernel = StructureBasedNeuralTangentKernel(K=2, L=3)
    diag_list = kernel.diag(features, graph.to_sparse())
    assert len(diag_list) == 2  # K vectors, independent of L
    torch.testing.assert_close(diag_list[0], (graph @ features @ features.T @ graph).diag().sqrt())

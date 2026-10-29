"""Clustering kernels preserve algorithms, budgets, ties and sparse semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans
from torch_sparse import SparseTensor

from graphslim.reduction.cluster_ops import (
    fitted_labels,
    nearest_squared,
    softmax_cluster_centers,
    sparse_class_assignment,
    sparse_membership,
    weighted_propagation,
)


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("budget", [16, 4096, 64 * 1024**2])
def test_nearest_exact_with_ties_and_both_chunk_dimensions(device, budget):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(4)
    points = torch.randn(113, 7, generator=generator).to(device)
    centers = torch.randn(17, 7, generator=generator).to(device)
    centers[1] = centers[0]
    points[0] = centers[0]
    expected = (points[:, None] - centers[None]).square().sum(2).min(1)
    values, indices = nearest_squared(points, centers, budget)
    assert torch.equal(indices, expected.indices)
    assert torch.equal(values, expected.values)
    assert indices[0].item() == 0


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans, BisectingKMeans])
@pytest.mark.parametrize("tied", [False, True])
def test_reusing_fitted_labels_matches_previous_predict(estimator, tied):
    features = np.random.default_rng(5).normal(size=(117, 9)).astype("float32")
    if tied:
        features = np.round(features)
    model = estimator(n_clusters=5, random_state=0, n_init=3)
    actual = fitted_labels(model, features)
    assert np.array_equal(actual, model.predict(features))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segmented_softmax_matches_masked_weighted_means(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(6)
    values = torch.randn(113, 17, generator=generator).to(device)
    scores = (torch.randn(113, generator=generator) * 100).to(device)
    groups = torch.arange(113, device=device) % 7
    expected = torch.stack(
        [
            (values[groups == group] * (scores[groups == group] / 0.7).softmax(0)[:, None]).sum(0)
            for group in range(7)
        ]
    )
    actual = softmax_cluster_centers(values, scores, groups, 7, 0.7)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_streamed_hops_equal_reference_and_two_spmm_equal_squared_operator():
    generator = torch.Generator().manual_seed(2)
    dense = torch.rand(19, 19, generator=generator)
    dense[dense < 0.8] = 0
    dense = dense / dense.sum(1, keepdim=True).clamp_min(1)
    adjacency = SparseTensor.from_dense(dense)
    features = torch.rand(19, 7, generator=generator)
    from torch_sparse import matmul

    hops = [features]
    for _ in range(3):
        hops.append(matmul(adjacency, hops[-1]).float())
    expected = torch.zeros_like(features)
    for weight, hop in zip([0.9, 0.1, 0, 0.5], hops):
        expected.add_(weight * hop)
    assert torch.equal(weighted_propagation(adjacency, features, [0.9, 0.1, 0, 0.5]), expected)
    torch.testing.assert_close(
        weighted_propagation(adjacency, features, [0, 0, 1]),
        matmul(matmul(adjacency, adjacency), features),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sparse_vng_membership_matches_dense_and_scales_with_samples():
    rng = np.random.default_rng(2)
    labels = np.arange(113) % 7
    weights = rng.integers(1, 10, size=113)
    dense = np.zeros((7, 113))
    dense[labels, np.arange(113)] = weights
    dense /= dense.sum(1, keepdims=True)
    actual = sparse_membership(labels, 7, weights, normalize=True)
    assert actual.nnz == len(labels)
    np.testing.assert_allclose(actual.toarray(), dense, atol=1e-15, rtol=1e-15)
    adjacency = sp.random(113, 113, density=0.1, random_state=2, format="csr")
    features = rng.normal(size=(113, 11))
    np.testing.assert_allclose(
        actual @ adjacency @ features, dense @ adjacency @ features, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("fuzzy", [False, True])
def test_gecc_sparse_assignment_uses_original_ids_and_normalizes(fuzzy):
    features = torch.tensor([[1.0, 0], [2.0, 0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    training = torch.tensor([[1.0, 0], [0.0, 1.0], [2.0, 0]])
    assignment = sparse_class_assignment(
        features,
        labels,
        training,
        torch.tensor([0, 1, 0]),
        torch.tensor([7, 3, 9]),
        12,
        fuzzy=fuzzy,
    )
    assert assignment.layout == torch.sparse_coo
    matrix = assignment.to_dense()
    torch.testing.assert_close(matrix.sum(1), torch.ones(3))
    assert not matrix[:, [0, 1, 2, 4, 5, 6, 8, 10, 11]].any()
    assert matrix[2, 3] == 1
    if fuzzy:
        assert matrix[0, 7] == matrix[1, 9] == 1


def test_gecc_chunked_lloyd_preserves_centers_and_empty_cluster_rng():
    from graphslim.condensation.gecc import perform_balance_evolve_clusteringjk

    points = torch.tensor([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [10.0, 1.0], [11.0, 1.0]])
    initial = torch.tensor([[0.0, 1.0], [0.0, 1.0], [10.0, 1.0]])
    expected = initial.clone()
    torch.manual_seed(5)
    for _ in range(4):
        assignments = (points[:, None] - expected[None]).square().sum(2).argmin(1)
        updated = expected.clone()
        for index in range(3):
            mask = assignments == index
            updated[index] = (
                points[mask].mean(0)
                if mask.any()
                else points[torch.randint(0, len(points), (1,)).item()]
            )
        shift = (updated - expected).square().sum().sqrt().item()
        expected = updated
        if shift < 1e-8:
            break
    args = SimpleNamespace(
        device="cpu",
        seed=5,
        fuzziness=1.0,
        rep_fuzz=1,
        balance_alpha=1,
        cluster_distance_working_bytes=32,
    )
    actual = perform_balance_evolve_clusteringjk(
        points, None, 0, initial, args, 3, epsilon=1e-8, max_iters=4
    )
    assert torch.equal(actual, expected)


def test_gecc_products_uses_full_sparse_propagation_and_real_reduction(tmp_path):
    from test_lightweight_methods import quick_args, tiny_graph
    from graphslim.condensation.gecc import GECC
    from graphslim.utils import seed_everything

    args = quick_args(tmp_path, "gecc", "trans")
    args.dataset = "ogbn-products"
    args.agg_alpha, args.agg_beta, args.agg_gamma = 0.1, 0.0, 0.9
    args.depth, args.fuzziness, args.rep_fuzz, args.balance_alpha = 2, 1.0, 1, 1
    graph = tiny_graph()
    seed_everything(3)
    actual = GECC("trans", graph, args).reduce(graph, verbose=False, save=False)
    assert actual.feat_syn.shape == (8, 8)
    assert torch.isfinite(actual.feat_syn).all()
    assert actual.assignment.layout == torch.sparse_coo
    args.dataset = "cora"
    seed_everything(3)
    other = tiny_graph()
    expected = GECC("trans", other, args).reduce(other, verbose=False, save=False)
    assert torch.equal(actual.feat_syn, expected.feat_syn)
    assert torch.equal(actual.labels_syn, expected.labels_syn)


def test_vng_teacher_and_sparse_compression_complete(tmp_path):
    from test_lightweight_methods import quick_args, tiny_graph
    from graphslim.coarsening.vng import VNG
    from graphslim.utils import seed_everything

    args = quick_args(tmp_path, "vng", "trans")
    seed_everything(13)
    graph = tiny_graph()
    result = VNG("trans", graph, args).reduce(graph, verbose=False)
    assert result.feat_syn.shape == (8, 8)
    assert result.adj_syn.shape == (8, 8)
    assert result.labels_syn.shape == (8,)
    assert torch.isfinite(result.feat_syn).all()
    assert torch.isfinite(result.adj_syn).all()

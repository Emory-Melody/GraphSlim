# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Numerical helpers shared by condensation reducers (gradient matching, spectra, curricula)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from graphslim.condensation import utils


def metric(name):
    return SimpleNamespace(dis_metric=name)


def test_match_loss_metrics_are_zero_for_identical_gradients_and_positive_otherwise():
    torch.manual_seed(0)
    real = [torch.randn(3, 4), torch.randn(4)]
    other = [torch.randn(3, 4), torch.randn(4)]
    for name in ("ours", "mse", "cos"):
        same = utils.match_loss(real, real, metric(name), "cpu")
        different = utils.match_loss(other, real, metric(name), "cpu")
        assert abs(float(same)) < 1e-5 and float(different) > 1e-3
    expected = sum(((a - b) ** 2).sum() for a, b in zip(other, real))
    torch.testing.assert_close(utils.match_loss(other, real, metric("mse"), "cpu"), expected)
    with pytest.raises(SystemExit):
        utils.match_loss(real, real, metric("unknown"), "cpu")


def test_distance_wb_handles_every_tensor_rank():
    torch.manual_seed(1)
    # Bias/normalization gradients are ignored by the layerwise distance.
    assert utils.distance_wb(torch.randn(5), torch.randn(5)) == 0
    for shape in ((3, 4), (2, 3, 4), (2, 3, 2, 2)):
        gradient = torch.randn(shape)
        assert float(utils.distance_wb(gradient, gradient)) < 1e-4
        opposite = utils.distance_wb(gradient, -gradient)
        torch.testing.assert_close(opposite, torch.tensor(2.0 * shape[0 if len(shape) != 2 else 1]))


def test_subgraph_and_knn_adjacency_builders():
    adjacency = torch.tensor([[0.0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    sub = utils.sub_E(torch.tensor([0, 1, 3]), adjacency).to_dense()
    torch.testing.assert_close(sub, torch.tensor([[0.0, 1, 1], [1, 0, 0], [1, 0, 0]]))
    points = torch.tensor([[0.0], [0.1], [5.0], [5.3]])
    knn = utils.update_E(points, 1.0).to_dense()
    torch.testing.assert_close(knn, knn.T)
    assert knn.diagonal().eq(1).all() and knn[0, 1] == 1 and knn[2, 3] == 0
    assert int(knn.sum()) == 6
    # An odd target edge count is rounded up to keep the pattern symmetric.
    assert int(utils.update_E(points, 1.5).to_dense().sum()) == 8


def test_normalization_and_graph_filter():
    data = torch.tensor([[1.0, 3.0], [3.0, 3.0]])
    normalized = utils.normalize_data(data)
    torch.testing.assert_close(normalized[:, 0], torch.tensor([-1.0, 1.0]) / math.sqrt(2))
    torch.testing.assert_close(normalized[:, 1], torch.zeros(2))
    adjacency = torch.ones(2, 2)
    features = torch.tensor([[2.0], [0.0]])
    torch.testing.assert_close(utils.GCF(adjacency, features, k=2), torch.tensor([[1.0], [1.0]]))


def difficulty_graph():
    # Node 0 has same-label neighbours only; node 2 sits between two classes.
    adjacency = torch.tensor([[0.0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]).to_sparse()
    labels = torch.tensor([0, 0, 1, 1])
    return SimpleNamespace(idx_train=np.array([0, 1, 2, 3])), adjacency, labels


def test_neighbourhood_difficulty_orders_homophilous_nodes_first():
    data, adjacency, labels = difficulty_graph()
    difficulty = utils.difficulty_measurer(data, adjacency, labels)
    assert difficulty[0] < 1e-6 and difficulty[1] > 0.5
    torch.testing.assert_close(difficulty, utils.difficulty_measurer_in(data, adjacency, labels))
    order = utils.sort_training_nodes(data, adjacency, labels)
    assert set(order[:2]) == {0, 3} and set(order[2:]) == {1, 2}
    assert set(utils.sort_training_nodes_in(data, adjacency, labels)[:2]) == {0, 3}


@pytest.mark.parametrize("scheduler", ["linear", "root", "geom"])
def test_training_schedulers_start_at_lambda_and_end_at_one(scheduler):
    assert utils.training_scheduler(0.25, 0, 10, scheduler) == pytest.approx(0.25)
    assert utils.training_scheduler(0.25, 10, 10, scheduler) == pytest.approx(1.0)
    middle = utils.training_scheduler(0.25, 5, 10, scheduler)
    assert 0.25 < middle < 1


def test_largest_component_and_training_map():
    # Component {0,1,2,3} is the largest; nodes 4 and 5 are a separate pair.
    rows = [0, 1, 1, 2, 2, 3, 4]
    cols = [1, 0, 2, 1, 3, 2, 5]
    adjacency = sp.csr_matrix((np.ones(7), (rows, cols)), shape=(6, 6))
    adjacency = adjacency.maximum(adjacency.T)
    idx_lcc, normalized, raw = utils.get_largest_cc(adjacency, 6, "toy")
    assert sorted(idx_lcc) == [0, 1, 2, 3] and normalized.shape == (4, 4)
    connected = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    idx_all, _, raw_all = utils.get_largest_cc(connected, 3, "toy")
    assert sorted(idx_all) == [0, 1, 2] and raw_all is connected
    same, mapping = utils.get_train_lcc(idx_all, np.array([0, 2]), torch.tensor([0, 1, 0]), 3, 2)
    assert same == [0, 2] and mapping == [idx_all.index(0), idx_all.index(2)]


@pytest.mark.parametrize("as_type", [torch.tensor, np.array, list])
def test_training_map_never_uses_held_out_nodes(as_type):
    # LCC listed out of order; training nodes 4 and 5 lie outside it.
    idx_lcc = [3, 0, 2, 1]
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    train_lcc, idx_map = utils.get_train_lcc(idx_lcc, as_type([2, 4, 0, 5]), labels, 6, 2)
    # Regression: torch indices used to give an empty overlap, and missing classes were
    # refilled from non-training nodes (here 1 and 3) using their labels.
    assert train_lcc == [2, 0]
    assert [idx_lcc[i] for i in idx_map] == train_lcc
    assert utils.node_id_list(torch.tensor([[3], [1]])) == [3, 1]


def test_spectral_selection_and_embeddings():
    values = torch.arange(6.0)
    vectors = torch.eye(6)
    low_high_values, low_high_vectors = utils.get_syn_eigen(values, vectors, 4, 0.5)
    assert low_high_values.tolist() == [0, 1, 4, 5]
    torch.testing.assert_close(low_high_vectors, vectors[:, [0, 1, 4, 5]])
    strided, _ = utils.get_syn_eigen(values, vectors, 2, 0.5, step=2)
    assert strided.tolist() == [0, 5]
    features = torch.arange(12.0).reshape(6, 2)
    basis = torch.eye(6)[:, :2]
    projected, per_vector = utils.get_subspace_embed(basis, features)
    torch.testing.assert_close(projected, features[:2])
    assert per_vector.shape == (2, 6, 2)
    torch.testing.assert_close(per_vector.sum(0)[:2], features[:2])
    covariance = utils.get_subspace_covariance_matrix(basis, features)
    assert covariance.shape == (2, 2, 2)
    torch.testing.assert_close(covariance.diagonal(dim1=1, dim2=2).sum(1), torch.ones(2))
    filtered = utils.get_embed_sum(torch.tensor([0.0, 1.0]), basis, features)
    torch.testing.assert_close(filtered[:2], torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    # Regression: an empty class (here class 1 of 3) yields a zero row, not NaN.
    empty = utils.get_embed_mean(torch.ones(2, 2), torch.tensor([0, 2]), num_classes=3)
    torch.testing.assert_close(empty[1], torch.zeros(2))
    assert torch.isfinite(empty).all() and empty.shape == (3, 2)
    means = utils.get_embed_mean(
        torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1, 1])
    )
    torch.testing.assert_close(
        means, torch.tensor([[0.6, 0.8], [1 / math.sqrt(5), 2 / math.sqrt(5)]])
    )


def test_eigendecomposition_cache_is_written_and_reused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    laplacian = sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    values, vectors = utils.get_eigh(laplacian, "toy", save=True)
    np.testing.assert_allclose(values, [0, 2], atol=1e-12)
    assert (tmp_path / "data" / "toy" / "eigenvectors.npy").exists()
    cached_values, _ = utils.get_eigh(None, "toy", save=False)
    np.testing.assert_allclose(cached_values, values)
    loaded_values, loaded_vectors = utils.load_eigen("toy", "./data")
    np.testing.assert_allclose(loaded_values, values)
    np.testing.assert_allclose(loaded_vectors, vectors)
    utils.get_eigh(np.diag([3.0, 1.0]), "unsaved", save=False)
    assert not (tmp_path / "data" / "unsaved" / "eigenvalues.npy").exists()
    # Regression: the cache root is configurable and matches load_eigen's load_path.
    root = tmp_path / "elsewhere"
    utils.get_eigh(laplacian, "toy2", save=True, root=str(root))
    np.testing.assert_allclose(utils.load_eigen("toy2", str(root))[0], values, atol=1e-12)
    assert not (tmp_path / "data" / "toy2").exists()
    # root=None computes without reading or writing any cache.
    np.save(tmp_path / "data" / "eigenvalues.npy", np.array([9.0, 9.0]))
    np.save(tmp_path / "data" / "eigenvectors.npy", np.eye(2))
    fresh, _ = utils.get_eigh(np.diag([3.0, 1.0]), "", save=True, root=None)
    assert fresh.tolist() == [1.0, 3.0]
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [
        "eigenvalues.npy",
        "eigenvectors.npy",
        "toy",
        "unsaved",
    ]


def test_large_dataset_spectra_merge_both_ends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def partial(A, k, which, tol):
        calls.append((k, which))
        return (
            (np.array([3.0, 2.0]), np.eye(3)[:, :2])
            if which == "LA"
            else (np.array([0.0]), np.eye(3)[:, 2:])
        )

    monkeypatch.setattr(utils, "eigsh", partial)
    values, vectors = utils.get_eigh(sp.eye(3).tocsr(), "flickr", save=True)
    assert calls == [(1000, "SA"), (1000, "LA")]
    assert values.tolist() == [0.0, 2.0, 3.0]
    np.testing.assert_array_equal(vectors[:, 0], [0, 0, 1])
    cached, _ = utils.get_eigh(None, "flickr", save=False)
    assert cached.tolist() == [0.0, 2.0, 3.0] and len(calls) == 2
    loaded, loaded_vectors = utils.load_eigen("flickr", "./data")
    assert loaded.tolist() == [0.0, 2.0, 3.0]
    np.testing.assert_array_equal(loaded_vectors, vectors)

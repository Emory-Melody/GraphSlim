# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""GECC clustering condensation: real reductions, centroid updates and error paths."""

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graphslim.condensation import gecc
from graphslim.dataset import load_reduced
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_legacy_reducer_integration import graph_fixture
from test_lightweight_methods import quick_args


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gecc_args(tmp_path, setting, fuzziness):
    args = quick_args(tmp_path, "gecc", setting)
    args.fuzziness, args.rep_fuzz, args.balance_alpha = fuzziness, 2, 1.0
    return args


def cluster_args(fuzziness=1.0, rep_fuzz=2, **extra):
    return SimpleNamespace(
        device="cpu",
        seed=3,
        fuzziness=fuzziness,
        rep_fuzz=rep_fuzz,
        balance_alpha=0.5,
        dataset="toy",
        si="1",
        **extra,
    )


def two_blobs():
    generator = torch.Generator().manual_seed(0)
    points = torch.cat(
        [torch.randn(6, 2, generator=generator), torch.randn(6, 2, generator=generator) + 10]
    )
    return points


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("fuzziness", [1.0, 1.3])
def test_gecc_reduces_to_class_balanced_centroids_and_roundtrips(tmp_path, setting, fuzziness):
    args = gecc_args(tmp_path, setting, fuzziness)
    graph = graph_fixture()
    seed_everything(2)
    reducer = create_reducer("gecc", setting, graph, args)
    reduced = reducer.reduce(graph, verbose=True)
    assert reduced.feat_syn.shape == (12, 8)
    assert reduced.labels_syn.tolist() == [0] * 6 + [1] * 6
    torch.testing.assert_close(reduced.adj_syn, torch.eye(12))
    assert torch.isfinite(reduced.feat_syn).all()
    # Every synthetic node is assigned only to training nodes of its own class.
    assignment = reduced.assignment.coalesce()
    assert assignment.shape == (12, 32)
    rows, columns = assignment.indices()
    assert set(columns.tolist()) <= set(graph.idx_train.tolist())
    torch.testing.assert_close(reduced.labels_syn[rows], graph.labels_full[columns])
    adjacency, features, labels = load_reduced(args)
    torch.testing.assert_close(features, reduced.feat_syn)
    assert labels.bincount().tolist() == [6, 6] and adjacency.shape == (12, 12)


def test_gecc_prepares_rare_class_first_labels_when_none_exist(tmp_path):
    args = gecc_args(tmp_path, "trans", 1.0)
    graph = graph_fixture()
    reducer = create_reducer("gecc", "trans", graph, args)
    graph.labels_train = torch.tensor([0] * 4 + [1] * 20)
    labels, train, slices = reducer.prepare_select(graph, args)
    assert labels.tolist() == [0, 0] + [1] * 10
    assert slices == {0: (0, 2), 1: (2, 12)}
    assert train is graph.labels_train
    graph = graph_fixture()
    reducer = create_reducer("gecc", "trans", graph, args)
    graph.labels_syn = None
    reduced = reducer.reduce(graph, verbose=True, save=False)
    assert reduced.labels_syn.bincount().tolist() == [6, 6]
    assert not (tmp_path / "reduced_graph").exists()


def test_gecc_skips_absent_classes_and_saves_empty_graph_without_any_class(tmp_path):
    args = gecc_args(tmp_path, "trans", 1.0)
    graph = graph_fixture()
    reducer = create_reducer("gecc", "trans", graph, args)
    graph.nclass = 3  # class 2 has no training nodes and must be skipped
    reduced = reducer.reduce(graph, verbose=True, save=False)
    assert reduced.labels_syn.unique().tolist() == [0, 1]
    graph = graph_fixture()
    reducer = create_reducer("gecc", "trans", graph, args)
    graph.nclass = 0
    assert reducer.reduce(graph, verbose=True) is graph
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (0, 8) and labels.numel() == 0 and adjacency.shape == (0, 0)


def test_gecc_rejects_wrong_centroid_counts(tmp_path, monkeypatch):
    args = gecc_args(tmp_path, "trans", 1.0)
    graph = graph_fixture()
    reducer = create_reducer("gecc", "trans", graph, args)
    monkeypatch.setattr(gecc, "perform_balance_evolve_clusteringjk", lambda **kw: None)
    with pytest.raises(ValueError, match="No centroids"):
        reducer.reduce(graph, verbose=False)
    monkeypatch.setattr(gecc, "perform_balance_evolve_clusteringjk", lambda **kw: torch.zeros(1, 8))
    with pytest.raises(ValueError, match="returned centroids"):
        reducer.reduce(graph, verbose=False)


def test_kmeans_separates_blobs_and_saves_partition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    points = two_blobs()
    centers = gecc.perform_balance_evolve_clusteringjk(
        points.numpy(), None, 0, None, cluster_args(), 2, save_p=True, class_id=4
    )
    assert centers.shape == (2, 2)
    means = sorted(c[0].item() for c in centers)
    assert means[0] < 3 < 7 < means[1]
    partition = np.load(tmp_path / "toy_1_4_partition_matrix.npy")
    assert partition.shape == (12, 2) and (partition.sum(1) == 1).all()
    assert sorted(partition.sum(0).tolist()) == [6, 6]


@pytest.mark.parametrize("fuzziness", [1.0, 1.5])
def test_previous_centroids_are_kept_extended_or_truncated(fuzziness):
    points = two_blobs()
    data = points if fuzziness == 1.0 else points.numpy()
    args = cluster_args(fuzziness, rep_fuzz=1)
    old = points[[0, 6]].clone()
    for previous, wanted in ((old, 2), (old[:1], 2), (old, 1)):
        centers = gecc.perform_balance_evolve_clusteringjk(data, None, 0, previous, args, wanted)
        assert centers.shape == (wanted, 2) and torch.isfinite(centers).all()
    centers = gecc.perform_balance_evolve_clusteringjk(data, None, 0, None, args, 2)
    assert sorted(round(c[0].item()) // 5 for c in centers) == [0, 2]


@pytest.mark.parametrize("fuzziness", [1.0, 1.5])
def test_clustering_rejects_too_few_points(fuzziness):
    points = two_blobs()[:2]
    data = points if fuzziness == 1.0 else points.numpy()
    args = cluster_args(fuzziness)
    with pytest.raises(ValueError, match="distinct centroids"):
        gecc.perform_balance_evolve_clusteringjk(data, None, 7, None, args, 3)
    with pytest.raises(ValueError, match="new centroids"):
        gecc.perform_balance_evolve_clusteringjk(data, None, 7, torch.zeros(1, 2), args, 4)


def test_empty_kmeans_cluster_is_reseeded():
    # Two coincident far-away centers leave one cluster empty on the first pass.
    points = torch.zeros(5, 2)
    args = cluster_args(rep_fuzz=1)
    centers = gecc.perform_balance_evolve_clusteringjk(
        points, None, 0, torch.full((2, 2), 9.0), args, 2
    )
    torch.testing.assert_close(centers, torch.zeros(2, 2))


def test_kmeanspp_initialization_spreads_centers():
    points = two_blobs()
    assert gecc.incremental_kmeanspp_init(points, None, 0, "cpu").shape == (0, 2)
    torch.manual_seed(0)
    new = gecc.incremental_kmeanspp_init(points, points[:1], 1, "cpu")
    assert new[0, 0] > 5  # far from the only old center in the first blob
    # Identical points have zero spread and fall back to uniform sampling.
    same = gecc.incremental_kmeanspp_init(torch.ones(4, 2), torch.ones(1, 2), 2, "cpu")
    torch.testing.assert_close(same, torch.ones(2, 2))


def test_fuzzy_helpers_normalize_memberships_and_penalize_imbalance(tmp_path, monkeypatch):
    points = two_blobs().double().numpy()
    data_t = points.T
    centers = points[[0, 6]]
    membership = gecc.init_membership_from_centers(centers, data_t, 2.0)
    assert membership.shape == (2, 12) and (membership >= 0).all()
    # Regression: each sample's memberships sum to one over the clusters.
    np.testing.assert_allclose(membership.sum(0), np.ones(12))
    assert (membership[0, :6] > 0.5).all() and (membership[1, 6:] > 0.5).all()
    # A sample located exactly on a center dominates that center's memberships.
    assert membership[0, 0] > 0.99 and membership[1, 6] > 0.99
    args = cluster_args(2.0, rep_fuzz=2)
    np.random.seed(0)
    obj, cntr, u = gecc.run_fuzzy_cmeans(0, args, data_t, 2, 2.0, 1e-6, 50, membership, 12)
    obj2, _, _ = gecc.run_fuzzy_cmeans(1, args, data_t, 2, 2.0, 1e-6, 50, None, 12)
    assert np.isfinite(obj) and np.isfinite(obj2)
    assert cntr.shape == (2, 2) and u.shape == (2, 12)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    best = gecc.parallel_fuzzy_cmeans(args, points, 2, 1e-6, 50, save_p=True, class_id=2)
    assert best.shape == (2, 2) and best.dtype == torch.float32
    assert sorted(round(c[0].item()) // 5 for c in best) == [0, 2]
    saved = np.load(tmp_path / "toy_1_2_partition_matrix.npy")
    np.testing.assert_allclose(saved.sum(0), np.ones(12))

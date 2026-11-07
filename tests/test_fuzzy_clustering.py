# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Actual hard/fuzzy optimization, warm starts, and saved memberships."""

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


def points():
    return np.array(
        [[-4.0, -4], [-3.9, -4.2], [-4.2, -3.8], [4, 4], [3.9, 4.2], [4.2, 3.8]], dtype=np.float32
    )


def options(fuzziness):
    return SimpleNamespace(
        device="cpu",
        seed=5,
        fuzziness=fuzziness,
        rep_fuzz=2,
        balance_alpha=0.1,
        dataset="tiny",
        si="1",
    )


@pytest.mark.parametrize("fuzziness", [1.0, 1.6])
@pytest.mark.parametrize("old_count", [0, 1, 2, 3])
def test_warm_start_cluster_resize_retains_two_separated_groups(
    tmp_path, monkeypatch, fuzziness, old_count
):
    monkeypatch.chdir(tmp_path)
    data = points()
    previous = torch.tensor([[-4.0, -4], [4, 4], [20, 20]])[:old_count] if old_count else None
    seed_everything(5)
    centers = gecc.perform_balance_evolve_clusteringjk(
        data, None, 0, previous, options(fuzziness), 2, max_iters=30, save_p=True
    )
    assert centers.shape == (2, 2) and torch.isfinite(centers).all()
    ordered = centers[centers[:, 0].argsort()]
    expected = torch.tensor(np.stack([data[:3].mean(0), data[3:].mean(0)]))
    torch.testing.assert_close(ordered, expected, atol=0.05, rtol=0)
    if fuzziness == 1:
        partition = np.load("tiny_1_0_partition_matrix.npy")
        np.testing.assert_array_equal(partition.sum(1), np.ones(len(data)))
        assert set(np.unique(partition)) == {0, 1}


@pytest.mark.parametrize("fuzziness", [1.0, 1.6])
def test_cluster_initialization_rejects_impossible_distinct_budgets(fuzziness):
    with pytest.raises(ValueError, match="Not enough data"):
        gecc.perform_balance_evolve_clusteringjk(points()[:1], None, 0, None, options(fuzziness), 2)


def test_fuzzy_cmeans_restarts_return_probability_memberships():
    data = points()
    centers = np.array([[-4.0, -4], [4, 4]])
    membership = gecc.init_membership_from_centers(centers, data.T, 1.6)
    assert membership.shape == (2, 6)
    assert (membership >= 0).all() and np.isfinite(membership).all()
    for seed in [0, 1]:
        objective, learned, partition = gecc.run_fuzzy_cmeans(
            seed, options(1.6), data.T, 2, 1.6, 1e-5, 30, membership, 6
        )
        assert np.isfinite(objective) and np.isfinite(learned).all()
        np.testing.assert_allclose(partition.sum(0), np.ones(6))
        ordered = learned[learned[:, 0].argsort()]
        np.testing.assert_allclose(ordered, [data[:3].mean(0), data[3:].mean(0)], atol=0.05)


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("fuzziness", [1.0, 1.6])
def test_gecc_reduction_outputs_finite_class_balanced_checkpoints(tmp_path, setting, fuzziness):
    args = quick_args(tmp_path, "gecc", setting)
    args.fuzziness, args.rep_fuzz, args.depth = fuzziness, 2, 3
    graph = graph_fixture()
    seed_everything(5)
    reduced = create_reducer("gecc", setting, graph, args).reduce(graph, verbose=True)
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (12, 8) and torch.isfinite(features).all()
    assert labels.bincount().tolist() == [6, 6]
    torch.testing.assert_close(adjacency, torch.eye(12))
    assignment = reduced.assignment.to_dense()
    assert assignment.shape == (12, graph.num_nodes)
    assert torch.isfinite(assignment).all()
    assert assignment[:, graph.idx_val].count_nonzero() == 0
    assert assignment[:, graph.idx_test].count_nonzero() == 0


def test_parallel_fuzzy_restarts_use_real_bounded_workers_and_save_memberships(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    executor = gecc.concurrent.futures.ProcessPoolExecutor
    monkeypatch.setattr(
        gecc.concurrent.futures, "ProcessPoolExecutor", lambda: executor(max_workers=2)
    )
    seed_everything(5)
    centers = gecc.parallel_fuzzy_cmeans(options(1.6), points(), 2, 1e-5, 30, save_p=True)
    assert centers.shape == (2, 2) and torch.isfinite(centers).all()
    membership = np.load("tiny_1_0_partition_matrix.npy")
    assert membership.shape == (2, 6)
    np.testing.assert_allclose(membership.sum(0), np.ones(6))

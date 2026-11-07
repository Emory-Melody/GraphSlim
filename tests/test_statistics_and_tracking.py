# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Incremental graph statistics, noise contracts, and offline tracking clients."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

from graphslim import tracking, utils
from graphslim.models.random_attack import RandomAttack


def test_incremental_degree_likelihood_matches_recomputing_each_graph():
    adjacency = torch.zeros(6, 6)
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (4, 5)]:
        adjacency[a, b] = adjacency[b, a] = 1
    pairs = torch.tensor([[0, 1], [0, 5], [1, 4]])
    minimum = torch.tensor(2.0)
    actual = utils.updated_log_likelihood_for_edge_changes(pairs, adjacency, minimum)
    for i, (a, b) in enumerate(pairs):
        changed = adjacency.clone()
        changed[a, b] = changed[b, a] = 1 - changed[a, b]
        expected = utils.degree_sequence_log_likelihood(changed.sum(1), minimum)
        for values, target in zip(actual, expected):
            torch.testing.assert_close(values[i], torch.as_tensor(target).float())
    np.testing.assert_array_equal(adjacency.sum(1), [3, 2, 3, 3, 2, 1])


def test_degree_statistics_agree_across_backends_and_ignore_below_threshold_nodes():
    degrees = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    numpy_result = utils.degree_sequence_log_likelihood(degrees, np.array(2.0))
    tensor_result = utils.degree_sequence_log_likelihood(
        torch.from_numpy(degrees), torch.tensor(2.0, dtype=torch.float64)
    )
    filtered_result = utils.degree_sequence_log_likelihood(degrees[degrees >= 2], np.array(2.0))
    for actual, tensor_value, filtered_value in zip(numpy_result, tensor_result, filtered_result):
        assert actual == pytest.approx(float(tensor_value))
        assert actual == pytest.approx(filtered_value)
    assert numpy_result[2] == 4


@pytest.mark.parametrize("test_only", [False, True])
@pytest.mark.parametrize("ratio", [0, 0.5, 1])
def test_feature_noise_delta_reconstructs_output_and_respects_scope(test_only, ratio):
    original = torch.ones(20, 3)
    data = SimpleNamespace(x=original.clone(), test_mask=torch.arange(20) >= 12)
    function = utils.add_feature_noise_test if test_only else utils.add_feature_noise
    delta, mask = function(data, ratio, seed=7)
    torch.testing.assert_close(data.x, original + delta)
    assert mask.sum() == int((8 if test_only else 20) * ratio)
    assert not delta[~mask].count_nonzero()
    if test_only:
        assert not mask[:12].any()
        torch.testing.assert_close(data.x[:12], original[:12])
    repeated = SimpleNamespace(x=original.clone(), test_mask=data.test_mask)
    repeated_delta, repeated_mask = function(repeated, ratio, seed=7)
    assert torch.equal(mask, repeated_mask) and torch.equal(delta, repeated_delta)


def test_constant_feature_rows_normalize_to_finite_zero_vectors():
    constant = torch.full((3, 4), 2.0)
    actual = utils.row_normalize_tensor(constant)
    torch.testing.assert_close(actual, torch.zeros_like(constant))
    varied = utils.row_normalize_tensor(torch.tensor([[1.0, 2], [3, 4]]))
    torch.testing.assert_close(varied.sum(1), torch.ones(2))


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "torch_sparse"])
def test_graph_logging_counts_edges_not_total_weight(kind):
    dense = torch.tensor([[0.0, 0.1, 0], [0.1, 0, 5], [0, 5, 0]])
    adjacency = {
        "dense": lambda: dense,
        "coo": dense.to_sparse,
        "csr": lambda: sp.csr_matrix(dense.numpy()),
        "torch_sparse": lambda: SparseTensor.from_dense(dense),
    }[kind]()
    summary = tracking.graph_summary(SimpleNamespace(feat_syn=torch.ones(3, 2), adj_syn=adjacency))
    assert summary == {"nodes": 3, "edges": 4, "density": 4 / 6}
    assert tracking.graph_summary(SimpleNamespace()) == {}


def test_tracking_client_logs_compact_config_and_finishes_without_network(monkeypatch):
    calls = []
    client = SimpleNamespace(
        init=lambda **kw: calls.append(("init", kw)),
        log=lambda metrics, **kw: calls.append(("log", metrics, kw)),
        finish=lambda: calls.append(("finish",)),
    )
    monkeypatch.setitem(sys.modules, "wandb", client)
    args = SimpleNamespace(
        wandb=True,
        wandb_project="test",
        wandb_run_name="run",
        count=2,
        logger=object(),
        _secret="hidden",
        callback=lambda: None,
    )
    value = tracking.build_tracker(args)
    value.log({"loss": 0.5}, step=3)
    value.log_graph(
        "reduced",
        SimpleNamespace(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]])),
        step=4,
    )
    value.finish()
    assert calls[0][1]["project"] == "test" and calls[0][1]["name"] == "run"
    assert "logger" not in calls[0][1]["config"] and "callback" not in calls[0][1]["config"]
    assert "_secret" not in calls[0][1]["config"]
    assert calls[1] == ("log", {"loss": 0.5}, {"step": 3})
    assert calls[2][1]["reduced/nodes"] == 3
    assert calls[-1] == ("finish",)
    assert tracking._compact_config(None) == {}


def test_optional_missing_tracker_falls_back_unless_required(monkeypatch):
    original = builtins.__import__

    def missing(name, *a, **kw):
        if name == "wandb":
            raise ImportError("test missing optional client")
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", missing)
    args = SimpleNamespace(wandb=True, wandb_required=False)
    assert isinstance(tracking.build_tracker(args), tracking.NullTracker)
    args.wandb_required = True
    with pytest.raises(RuntimeError, match="not installed"):
        tracking.build_tracker(args)


@pytest.mark.parametrize("mode", ["add", "remove", "flip"])
def test_random_edge_attacks_preserve_simple_undirected_graph_and_exact_budget(mode):
    adjacency = sp.diags([np.ones(9), np.ones(9)], [-1, 1], shape=(10, 10), format="csr")
    original = adjacency.toarray().copy()
    np.random.seed(8)
    attack = RandomAttack()
    attack.attack(adjacency, 3, type=mode)
    changed = attack.modified_adj.toarray()
    np.testing.assert_array_equal(changed, changed.T)
    np.testing.assert_array_equal(changed.diagonal(), np.zeros(10))
    assert np.count_nonzero(changed != original) == 6
    np.testing.assert_array_equal(adjacency.toarray(), original)
    if mode == "add":
        assert changed.sum() == original.sum() + 6
    elif mode == "remove":
        assert changed.sum() == original.sum() - 6


def test_random_feature_attack_preserves_input_and_erases_selected_columns():
    original = torch.ones(6, 5)
    attack = RandomAttack(attack_structure=False, attack_features=True)
    attack.attack(original, 2)
    assert (attack.modified_features.sum(0) == 0).sum() == 2
    assert original.eq(1).all()
    with pytest.raises(NotImplementedError):
        attack.inject_nodes(sp.eye(4), 1, 1)


def test_incremental_degree_likelihood_accepts_numpy_pairs():
    adjacency = torch.zeros(4, 4)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        adjacency[a, b] = adjacency[b, a] = 1
    pairs = np.array([[0, 2], [1, 2]])
    new_ll, alpha, n, _ = utils.updated_log_likelihood_for_edge_changes(
        pairs, adjacency, torch.tensor(1.0)
    )
    assert new_ll.shape == alpha.shape == n.shape == (2,)
    # Adding edge (0, 2) keeps four nodes at degree >= 1; removing (1, 2) does too.
    assert n.tolist() == [4.0, 4.0]


def test_row_normalization_shifts_by_global_minimum_and_sums_rows_to_one():
    matrix = torch.tensor([[1.0, 2], [3, 4]])
    varied = utils.row_normalize_tensor(matrix)
    torch.testing.assert_close(varied, torch.tensor([[0.0, 1.0], [2 / 5, 3 / 5]]))
    torch.testing.assert_close(varied.sum(1), torch.ones(2))
    # The global-minimum shift is applied to the caller's tensor in place.
    torch.testing.assert_close(matrix, torch.tensor([[0.0, 1], [2, 3]]))

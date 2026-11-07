# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Coverage for the reducer registry, lightweight reducer I/O and clustering operators."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

from graphslim.reduction import cluster_ops, lightweight, registry


# --------------------------------------------------------------------------- registry


def test_method_spec_loads_aggregation_variant_only_when_registered():
    spec = registry.MethodSpec(
        "x", "condensation", "json", "JSONDecoder", agg_module="json", agg_class_name="JSONEncoder"
    )
    assert spec.load_class() is json.JSONDecoder
    assert spec.load_class(use_agg=True) is json.JSONEncoder
    plain = registry.MethodSpec("y", "condensation", "json", "JSONDecoder")
    assert plain.load_class(use_agg=True) is json.JSONDecoder


def test_registry_normalizes_names_and_reports_unknown_methods():
    assert registry.normalize_method_name("  Heavy-Edge ") == "heavy_edge"
    assert registry.get_method_spec("algebraic_JC").name == "algebraic_jc"
    assert registry.get_method_spec("TSpanner").name == "t_spanner"
    with pytest.raises(ValueError, match="Unknown graph reduction method 'nope'"):
        registry.get_method_spec("nope")
    assert set(registry.list_methods("coarsening")) < set(registry.list_methods())


def test_create_reducer_passes_configuration_to_registered_class():
    args = SimpleNamespace(device="cpu", agg=False)
    reducer = registry.create_reducer("cgc", setting="trans", data=None, args=args)
    assert type(reducer).__name__ == "CGC" and reducer.setting == "trans"
    assert reducer.args is args


# --------------------------------------------------------------------------- lightweight helpers


DENSE = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.float32)


@pytest.mark.parametrize(
    "convert",
    [
        lambda: sp.coo_matrix(DENSE),
        lambda: torch.tensor(DENSE),
        lambda: torch.tensor(DENSE).to_sparse(),
        lambda: torch.tensor(DENSE).to_sparse_csr(),
        lambda: SparseTensor.from_dense(torch.tensor(DENSE)),
        lambda: DENSE.tolist(),
    ],
)
def test_as_csr_accepts_every_adjacency_representation(convert):
    matrix = lightweight.as_csr(convert())
    assert sp.isspmatrix_csr(matrix) and matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix.toarray(), DENSE)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_as_csr_detaches_sparse_tensors_that_require_grad(layout):
    """Regression: sparse tensors with autograd history could not be converted."""
    dense = torch.tensor(DENSE, requires_grad=True)
    sparse = dense.to_sparse() if layout == "coo" else dense.to_sparse_csr()
    assert sparse.requires_grad
    np.testing.assert_array_equal(lightweight.as_csr(sparse).toarray(), DENSE)
    np.testing.assert_array_equal(lightweight.as_csr(dense).toarray(), DENSE)


def test_as_csr_copies_scipy_input():
    source = sp.csr_matrix(DENSE)
    lightweight.as_csr(source).data[:] = 0
    assert source.sum() == 4


def test_normalized_adjacency_matches_closed_form_with_isolated_node():
    normalized = lightweight.normalized_adjacency(DENSE).to_dense()
    expected = torch.tensor([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0], [0, 0, 1]])
    torch.testing.assert_close(normalized, expected)
    sparse = lightweight.sparse_tensor(DENSE)
    assert sparse.is_sparse and sparse.is_coalesced() and sparse._nnz() == 2


def test_propagation_and_truncated_ppr_series():
    adjacency = torch.tensor([[0.0, 1], [1, 0]])
    features = torch.tensor([[1.0], [3.0]])
    torch.testing.assert_close(lightweight.propagate(adjacency, features, 0), features)
    torch.testing.assert_close(lightweight.propagate(adjacency, features, 3), features.flip(0))
    result = lightweight.denoise(adjacency, features, 2, 0.5)
    torch.testing.assert_close(result, 0.5 * features + 0.5 * 0.5 * features.flip(0))
    for steps, alpha in [(0, 0.5), (2, 1.0), (2, -0.1)]:
        with pytest.raises(ValueError, match="Denoising"):
            lightweight.denoise(adjacency, features, steps, alpha)


def test_class_budget_keeps_every_class_and_respects_target():
    labels = torch.tensor([0] * 6 + [1] * 3 + [2])
    budget = lightweight.class_budget(labels, 0.5)
    assert budget == {2: 1, 1: 1, 0: 3} and sum(budget.values()) == 5
    assert lightweight.class_budget(labels, 0.01) == {2: 1, 1: 1, 0: 1}
    assert lightweight.class_budget(labels, 1) == {2: 1, 1: 3, 0: 6}
    for bad_labels, rate in [(torch.tensor([], dtype=torch.long), 0.5), (labels, 0), (labels, 2)]:
        with pytest.raises(ValueError, match="reduction_rate"):
            lightweight.class_budget(bad_labels, rate)


def test_cluster_handles_trivial_sizes_and_rejects_invalid_counts():
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_array_equal(lightweight.cluster(features, 1, 0), np.zeros(4))
    np.testing.assert_array_equal(lightweight.cluster(features, 4, 0), np.arange(4))
    for size in (0, 5):
        with pytest.raises(ValueError, match="Cluster count"):
            lightweight.cluster(features, size, 0)


@pytest.mark.parametrize("minibatch", [False, True])
def test_cluster_separates_well_spaced_groups(minibatch):
    features = np.array([[0, 0], [0, 0.1], [10, 10], [10, 10.1], [20, 0], [20, 0.1]])
    labels = lightweight.cluster(features, 3, seed=3, minibatch=minibatch)
    assert labels.dtype == np.int64 and sorted(np.bincount(labels)) == [2, 2, 2]
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[4] == labels[5]


def test_cluster_repairs_empty_clusters_with_farthest_donor_member(monkeypatch):
    features = np.array([[0.0], [1.0], [2.0], [9.0], [10.0]])
    monkeypatch.setattr(
        lightweight, "fitted_labels", lambda estimator, x: np.array([0, 0, 0, 1, 1])
    )
    labels = lightweight.cluster(features, 4, seed=0)
    assert np.bincount(labels, minlength=4).min() == 1
    # Each empty cluster takes the member farthest from the largest donor's center
    # (first index on ties): node 0 from {0, 1, 2}, then node 1 from {1, 2}.
    assert labels.tolist() == [2, 3, 0, 1, 1]


# --------------------------------------------------------------------------- LightweightReducer


def _graph():
    adjacency = sp.csr_matrix(np.eye(4, k=1) + np.eye(4, k=-1), dtype=np.float32)
    features = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    return SimpleNamespace(
        feat_full=features,
        adj_full=adjacency,
        idx_train=torch.tensor([1, 3]),
        labels_train=torch.tensor([0, 1]),
        feat_train=features[[1, 3]],
        adj_train=adjacency[[1, 3]][:, [1, 3]],
    )


def _args(tmp_path, **overrides):
    args = SimpleNamespace(
        device="cpu",
        reduction_rate=0.5,
        method="cgc",
        save_path=str(tmp_path),
        dataset="toy",
        seed=4,
        attack=None,
        logger=logging.getLogger("cov-lightweight"),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_reducer_inputs_select_transductive_and_inductive_views(tmp_path):
    with pytest.raises(ValueError, match="setting"):
        lightweight.LightweightReducer("both", None, _args(tmp_path))
    graph = _graph()
    features, adjacency, indices, labels = lightweight.LightweightReducer(
        "trans", graph, _args(tmp_path)
    ).inputs(graph)
    assert features.shape == (4, 2) and adjacency.shape == (4, 4)
    assert indices.tolist() == [1, 3] and labels.tolist() == [0, 1]
    features, adjacency, indices, labels = lightweight.LightweightReducer(
        "ind", graph, _args(tmp_path)
    ).inputs(graph)
    assert features.tolist() == [[2.0, 3.0], [6.0, 7.0]] and indices.tolist() == [0, 1]
    assert adjacency.nnz == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda g, a: setattr(g, "labels_train", torch.tensor([0])), "single-class"),
        (lambda g, a: setattr(g, "labels_train", torch.ones(2, 2)), "single-class"),
        (lambda g, a: g.feat_full.__setitem__((0, 0), float("nan")), "finite"),
        (lambda g, a: setattr(a, "reduction_rate", 1.5), "reduction_rate"),
    ],
)
def test_reducer_inputs_reject_invalid_training_data(tmp_path, change, message):
    graph, args = _graph(), _args(tmp_path)
    change(graph, args)
    with pytest.raises(ValueError, match=message):
        lightweight.LightweightReducer("trans", graph, args).inputs(graph)


def test_reducer_budget_limit_is_configurable(tmp_path):
    reducer = lightweight.LightweightReducer("trans", None, _args(tmp_path, max_syn_nodes=3))
    reducer.check_budget(3)
    with pytest.raises(ValueError, match="max_syn_nodes=3"):
        reducer.check_budget(4)
    lightweight.LightweightReducer("trans", None, _args(tmp_path)).check_budget(4096)


def test_reducer_finish_validates_saves_and_reports(tmp_path, capsys):
    args = _args(tmp_path)
    reducer = lightweight.LightweightReducer("ind", None, args)
    data = SimpleNamespace()
    features, adjacency, labels = torch.ones(2, 3), torch.eye(2), torch.tensor([0, 1])
    result = reducer.finish(data, features, adjacency, labels, save=True, verbose=True, clusters=2)
    assert result is data and data.reduction_metadata["synthetic_nodes"] == 2
    assert data.reduction_metadata["clusters"] == 2
    assert data.reduction_metadata["uses_test_labels"] is False
    assert "'method': 'cgc'" in capsys.readouterr().out
    saved = tmp_path / "reduced_graph/cgc"
    torch.testing.assert_close(torch.load(saved / "adj_toy_0.5_4.pt"), torch.eye(2))
    torch.testing.assert_close(torch.load(saved / "feat_toy_0.5_4.pt"), features)
    with pytest.raises(ValueError, match="dimensions"):
        reducer.finish(data, features, torch.eye(3), labels, save=False, verbose=False)
    with pytest.raises(ValueError, match="dimensions"):
        reducer.finish(data, features[:0], torch.eye(0), labels[:0], save=False, verbose=False)
    with pytest.raises(FloatingPointError, match="non-finite"):
        reducer.finish(data, features * float("inf"), adjacency, labels, save=False, verbose=False)


# --------------------------------------------------------------------------- cluster_ops


@pytest.mark.parametrize(
    ("points", "centers", "budget", "message"),
    [
        (torch.ones(3), torch.ones(2, 1), 64, "two-dimensional"),
        (torch.ones(3, 2), torch.ones(2, 3), 64, "two-dimensional"),
        (torch.ones(3, 2), torch.ones(0, 2), 64, "Nonempty"),
        (torch.ones(3, 0), torch.ones(1, 0), 64, "Nonempty"),
        (torch.ones(3, 2), torch.ones(1, 2), 0, "Nonempty"),
        (torch.ones(3, 2), torch.ones(1, 2, dtype=torch.float64), 64, "dtype"),
    ],
)
def test_nearest_squared_validates_inputs(points, centers, budget, message):
    with pytest.raises(ValueError, match=message):
        cluster_ops.nearest_squared(points, centers, max_working_bytes=budget)


def test_softmax_centers_and_membership_validation():
    features = torch.tensor([[1.0, 0], [3.0, 2], [5.0, 5]])
    labels = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match="temperature"):
        cluster_ops.softmax_cluster_centers(features, torch.zeros(3), labels, 2, 0)
    centers = cluster_ops.softmax_cluster_centers(features, torch.zeros(3), labels, 2, 2.0)
    torch.testing.assert_close(centers, torch.tensor([[2.0, 1], [5.0, 5]]))
    matrix = cluster_ops.sparse_membership([0, 0, 2], 3, weights=[1, 3, 2], normalize=False)
    np.testing.assert_array_equal(matrix.toarray(), [[1, 3, 0], [0, 0, 0], [0, 0, 2]])
    with pytest.raises(ValueError, match="empty cluster"):
        cluster_ops.sparse_membership([0, 0, 2], 3, normalize=True)


def test_sparse_class_assignment_skips_classes_without_training_nodes():
    synthetic = torch.tensor([[0.0], [1.0]])
    assignment = cluster_ops.sparse_class_assignment(
        synthetic,
        torch.tensor([0, 2]),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([0, 0]),
        torch.tensor([4, 7]),
        original_nodes=8,
    )
    dense = assignment.to_dense()
    assert dense.shape == (2, 8)
    torch.testing.assert_close(dense[0, [4, 7]], torch.tensor([0.5, 0.5]))
    assert dense[1].sum() == 0
    empty = cluster_ops.sparse_class_assignment(
        synthetic, torch.tensor([3, 3]), synthetic, torch.tensor([0, 0]), [0, 1], 2
    )
    assert empty._nnz() == 0 and empty.shape == (2, 2)

"""Input and output contracts for independently implemented reducer extensions."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from graphslim.reduction.lightweight import (
    LightweightReducer,
    as_csr,
    class_budget,
    cluster,
    denoise,
)
from test_lightweight_methods import quick_args, tiny_graph


@pytest.mark.parametrize("layout", ["numpy", "dense", "coo", "csr", "torch_sparse"])
def test_adjacency_adapter_preserves_weights_for_differentiable_and_numpy_inputs(layout):
    dense = torch.tensor([[0.0, 0.25], [0.25, 1.0]], requires_grad=True)
    # Sparse layouts are built from detached values: as_csr cannot convert sparse tensors
    # that require grad (see the coverage report for this package limitation).
    value = {
        "numpy": lambda: dense.detach().numpy(),
        "dense": lambda: dense,
        "coo": dense.detach().to_sparse,
        "csr": dense.detach().to_sparse_csr,
        "torch_sparse": lambda: SparseTensor.from_dense(dense.detach()),
    }[layout]()
    actual = as_csr(value)
    np.testing.assert_array_equal(actual.toarray(), dense.detach().numpy())
    assert actual.dtype == np.float32


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("invalid", ["empty", "multilabel", "length", "nonfinite", "rate"])
def test_extension_inputs_reject_invalid_graphs_before_reduction(tmp_path, setting, invalid):
    graph = tiny_graph()
    args = quick_args(tmp_path, "cgc", setting)
    if invalid == "empty":
        graph.labels_train = torch.empty(0, dtype=torch.long)
    elif invalid == "multilabel":
        graph.labels_train = torch.ones(16, 2)
    elif invalid == "length":
        graph.labels_train = graph.labels_train[:-1]
    elif invalid == "nonfinite":
        graph.feat_full[0, 0] = graph.feat_train[0, 0] = float("nan")
    else:
        args.reduction_rate = 0
    reducer = LightweightReducer(setting, graph, args)
    with pytest.raises(ValueError):
        reducer.inputs(graph)


@pytest.mark.parametrize("invalid", ["empty", "adjacency", "labels", "nan"])
def test_extension_output_rejects_invalid_dimensions_or_nonfinite_values(tmp_path, invalid):
    args = quick_args(tmp_path, "cgc", "trans")
    reducer = LightweightReducer("trans", None, args)
    features, adjacency, labels = torch.ones(3, 2), torch.eye(3), torch.tensor([0, 1, 0])
    if invalid == "empty":
        features, adjacency, labels = features[:0], adjacency[:0, :0], labels[:0]
    elif invalid == "adjacency":
        adjacency = torch.eye(2)
    elif invalid == "labels":
        labels = labels[:-1]
    else:
        features[0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        reducer.finish(
            deepcopy(tiny_graph()), features, adjacency, labels, save=False, verbose=False
        )


def test_extension_setting_and_dense_adjacency_memory_budget_are_checked(tmp_path):
    args = quick_args(tmp_path, "cgc", "trans")
    with pytest.raises(ValueError, match="setting"):
        LightweightReducer("unknown", None, args)
    reducer = LightweightReducer("trans", None, args)
    args.max_syn_nodes = 3
    reducer.check_budget(3)
    with pytest.raises(ValueError, match="max_syn_nodes"):
        reducer.check_budget(4)


@pytest.mark.parametrize("count", [1, 4])
def test_clustering_endpoint_budgets_are_deterministic_without_empty_clusters(count):
    values = np.arange(8).reshape(4, 2)
    groups = cluster(values, count, 4)
    assert len(set(groups)) == count and len(groups) == 4
    with pytest.raises(ValueError):
        cluster(values, 5, 4)


@pytest.mark.parametrize("steps,alpha", [(0, 0.5), (2, -0.1), (2, 1)])
def test_denoising_rejects_invalid_iterations_or_damping(steps, alpha):
    with pytest.raises(ValueError):
        denoise(torch.eye(3), torch.ones(3, 2), steps, alpha)
    with pytest.raises(ValueError):
        class_budget(torch.tensor([], dtype=torch.long), 0.5)

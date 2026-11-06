"""Actual partial checkpoint recovery and downstream adjacency transformations."""

from types import SimpleNamespace

import pytest
import torch

from graphslim.condensation.utils import update_E
from graphslim.dataset.utils import get_syn_data, load_reduced, save_reduced, sparsify
from test_lightweight_methods import quick_args, tiny_graph


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("kind", ["feat", "label", "adj"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_partial_checkpoints_recover_only_the_unavailable_component(
    tmp_path, setting, kind, corrupt
):
    args = quick_args(tmp_path, "random", setting)
    graph = tiny_graph()
    features = graph.feat_full if setting == "trans" else graph.feat_train
    adjacency = torch.eye(len(features))
    labels = graph.labels_train
    save_reduced(adjacency, features, labels, args)
    file = (
        tmp_path
        / "reduced_graph/random"
        / f"{kind}_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
    )
    if corrupt:
        file.write_bytes(b"incomplete tensor checkpoint")
    else:
        file.unlink()
    actual_adj, actual_features, actual_labels = load_reduced(args, graph)
    torch.testing.assert_close(actual_adj, adjacency)
    torch.testing.assert_close(actual_features, features)
    torch.testing.assert_close(actual_labels, labels)


@pytest.mark.parametrize("device", [None, torch.device("cpu")])
def test_checkpoint_loader_accepts_optional_and_torch_device_objects(tmp_path, device):
    args = quick_args(tmp_path, "random", "trans")
    save_reduced(torch.eye(3), torch.ones(3, 2), torch.tensor([0, 1, 0]), args)
    args.device = device
    adj, features, labels = load_reduced(args)
    assert all(value.device.type == "cpu" for value in [adj, features, labels])


@pytest.mark.parametrize(
    "method,dataset,model,threshold",
    [
        ("gcond", "cora", "GAT", 0.5),
        ("doscond", "cora", "GAT", 0.5),
        ("gcond", "flickr", "GAT", 0.1),
        ("msgc", "cora", "GAT", 0.3),
        ("random", "cora", "GAT", 0.5),
        ("gcond", "cora", "GCN", 0.3),
        ("doscond", "cora", "GCN", 0.3),
        ("msgc", "cora", "GCN", 0.0),
    ],
)
def test_model_specific_sparsification_preserves_exact_threshold_boundary(
    method, dataset, model, threshold
):
    adjacency = torch.tensor([[0.0, 0.1, 0.3], [0.1, 0, 0.5], [0.3, 0.5, 0]])
    args = SimpleNamespace(method=method, dataset=dataset, threshold=0.3)
    actual = sparsify(model, adjacency.clone(), args, verbose=True)
    expected = adjacency.clone()
    expected[expected < threshold] = 0
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_reloaded_mlp_graph_uses_identity_without_modifying_saved_topology(tmp_path, sparse):
    args = quick_args(tmp_path, "gcond", "trans")
    graph = tiny_graph()
    adjacency = torch.ones(4, 4) - torch.eye(4)
    save_reduced(
        adjacency.to_sparse() if sparse else adjacency,
        torch.ones(4, 8),
        torch.tensor([0, 1, 0, 1]),
        args,
    )
    features, actual, labels = get_syn_data(graph, args, "MLP")
    torch.testing.assert_close(actual, torch.eye(4))
    torch.testing.assert_close(load_reduced(args)[0].to_dense(), adjacency)
    assert features.shape == (4, 8) and labels.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("neighbours", [0.0, 0.5])
def test_feature_distance_graph_selects_loops_and_nearest_symmetric_pairs(neighbours):
    features = torch.tensor([[0.0], [1.0], [4.0], [10.0]])
    actual = update_E(features, neighbours).coalesce().to_dense()
    # Zero neighbours yields loops only; a positive tiny budget retains the
    # uniquely closest pair in both directions.
    expected = torch.eye(4)
    if neighbours:
        expected[0, 1] = expected[1, 0] = 1
    torch.testing.assert_close(actual, expected)
    assert torch.equal(features, torch.tensor([[0.0], [1.0], [4.0], [10.0]]))

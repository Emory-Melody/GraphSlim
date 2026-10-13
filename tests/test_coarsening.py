from types import SimpleNamespace
import pytest
import scipy.sparse as sp
import torch

from graphslim.coarsening import Average, Cluster, ClusterAgg


@pytest.mark.parametrize("method", [Average, Cluster, ClusterAgg])
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_classwise_coarsening_handles_tensor_masks_and_reset_labels(method, setting):
    graph = SimpleNamespace(
        nclass=2,
        labels_train=torch.tensor([0, 0, 1, 1]),
        feat_train=torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]]),
        labels_syn=None,
        adj_full=sp.eye(4, format="csr"),
        adj_train=sp.eye(4, format="csr"),
        train_mask=torch.ones(4, dtype=torch.bool),
    )
    graph.feat_full = graph.feat_train
    args = SimpleNamespace(method="test", reduction_rate=0.5, device="cpu", setting=setting)
    result = method(setting, graph, args).reduce(graph, verbose=False, save=False)
    assert torch.equal(result.labels_syn, torch.tensor([0, 1]))
    assert torch.allclose(result.feat_syn, torch.tensor([[2.0, 3.0], [20.0, 30.0]]))
    assert torch.equal(result.adj_syn, torch.eye(2))


def test_cluster_fills_empty_cluster_slots_for_identical_features():
    graph = SimpleNamespace(
        nclass=1, feat_train=torch.ones(4, 2), labels_train=torch.zeros(4, dtype=torch.long)
    )
    args = SimpleNamespace(method="clustering", reduction_rate=0.5, device="cpu")
    result = Cluster("trans", graph, args).reduce(graph, verbose=False, save=False)
    assert result.feat_syn.shape == (2, 2)
    assert torch.isfinite(result.feat_syn).all()

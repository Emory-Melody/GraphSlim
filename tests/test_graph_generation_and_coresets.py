"""Actual graph-generator and coreset initialization paths on tiny graphs."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graphslim.models.parametrized_adj import PGE
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_legacy_reducer_integration import graph_fixture
from test_lightweight_methods import quick_args


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


# Two or three nodes would split the edge pairs into five chunks with single-row BatchNorm
# batches in training mode; realistic synthetic graphs are far larger.
@pytest.mark.parametrize("nodes", [4, 6])
def test_chunked_graph_generator_matches_full_evaluation_and_supports_small_training_batches(nodes):
    args = SimpleNamespace(dataset="reddit", reduction_rate=0.01)
    torch.manual_seed(4)
    model = PGE(3, nodes, device="cpu", args=args)
    full = deepcopy(model)
    full.args.dataset = "cora"
    model.eval()
    full.eval()
    features = torch.rand(nodes, 3, requires_grad=True)
    chunked = model(features)
    torch.testing.assert_close(chunked, full(features))
    torch.testing.assert_close(chunked, chunked.T)
    torch.testing.assert_close(chunked.diagonal(), torch.zeros(nodes))
    assert ((chunked >= 0) & (chunked <= 1)).all()
    chunked.sum().backward()
    assert torch.isfinite(features.grad).all()
    torch.testing.assert_close(model.inference(features), chunked.detach())
    model.train()
    assert torch.isfinite(model(features.detach())).all()


def test_arxiv_graph_generator_selects_dataset_width_and_retains_zero_diagonal():
    model = PGE(3, 4, device="cpu", args=SimpleNamespace(dataset="ogbn-arxiv", reduction_rate=0.5))
    assert model.layers[0].out_features == 256
    result = model(torch.rand(4, 3))
    assert result.shape == (4, 4) and result.diagonal().eq(0).all()


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_geom_coreset_trains_real_initializer_and_reloads_selected_graph(
    tmp_path, monkeypatch, setting
):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    args = quick_args(tmp_path / "checkpoints", "geom", setting)
    args.lr_coreset, args.coreset_epochs, args.condense_model = 0.01, 2, "GCN"
    graph = graph_fixture()
    seed_everything(5)
    reducer = create_reducer("geom", setting, graph, args)
    reducer.buf_dir = str(tmp_path / "buffer")
    (tmp_path / "buffer").mkdir()
    selected = reducer.init_coreset_select(graph)
    assert len(selected) == len(set(selected)) == 12
    features = graph.feat_full if setting == "trans" else graph.feat_train
    adjacency = graph.adj_full if setting == "trans" else graph.adj_train
    labels = graph.labels_full if setting == "trans" else graph.labels_train
    actual_features, actual_adjacency, actual_labels = reducer.get_coreset_init(
        features, adjacency, labels
    )
    np.testing.assert_array_equal(actual_features, features[selected].numpy())
    np.testing.assert_array_equal(
        actual_adjacency.toarray(), adjacency[np.ix_(selected, selected)].toarray()
    )
    torch.testing.assert_close(actual_labels, labels[selected])
    assert actual_labels.bincount().tolist() == [6, 6]

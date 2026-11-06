"""Real small-graph reductions, checkpoint reload, and downstream optimization."""

import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset import load_reduced, loader
from graphslim.dataset.utils import splits
from graphslim.evaluation import Evaluator
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph_fixture():
    generator = torch.Generator().manual_seed(42)
    nodes = torch.arange(32)
    graph = Data(
        x=torch.rand(32, 8, generator=generator),
        y=nodes % 2,
        edge_index=torch.stack([nodes.repeat(2), torch.cat([nodes.roll(1), nodes.roll(-1)])]),
        train_mask=nodes < 24,
        val_mask=(nodes >= 24) & (nodes < 28),
        test_mask=nodes >= 28,
    )
    result = loader.TransAndInd(splits(graph), "cora", norm=False)
    result.nclass = 2
    return result


@pytest.mark.parametrize("method", ["gcond", "doscond", "doscondx", "gcdm", "gcdmx"])
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_condensation_reduces_saves_and_trains_an_actual_gcn(
    tmp_path, monkeypatch, method, setting
):
    args = quick_args(tmp_path, method, setting)
    args.condense_model, args.nlayers = "GCN", 2
    args.outer_loop, args.inner_loop = 1, 1
    args.epochs, args.eval_epochs, args.run_inter_eval = 2, 3, 1
    args.checkpoints = [1]
    if method.startswith("gcdm"):
        args.dis_metric = "mse"
    # Only prevent worker processes in the external neighbor-sampling boundary.
    sampler = loader.NeighborSampler
    monkeypatch.setattr(
        loader, "NeighborSampler", lambda *a, **kw: sampler(*a, **dict(kw, num_workers=0))
    )
    graph = graph_fixture()
    seed_everything(4)
    reduced = create_reducer(method, setting, graph, args).reduce(graph, verbose=False)
    # Some validation draws can score zero. Save the output contract in that case
    # through the same public serializer rather than replacing validation scores.
    from graphslim.dataset.utils import save_reduced

    save_reduced(reduced.adj_syn, reduced.feat_syn, reduced.labels_syn, args)
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (12, 8) and adjacency.shape == (12, 12)
    assert labels.dtype == torch.long and labels.bincount().tolist() == [6, 6]
    assert torch.isfinite(features).all() and torch.isfinite(adjacency).all()
    torch.testing.assert_close(adjacency, adjacency.T)
    validation, test = Evaluator(args).test(
        reduced, "GCN", reduced=True, mode="cross", verbose=False
    )
    assert 0 <= validation <= 1 and 0 <= test <= 1


@pytest.mark.parametrize("method", ["variation_neighborhoods"])
def test_spectral_coarsening_pipeline_preserves_class_pure_components(tmp_path, method):
    graph = graph_fixture()
    # Each connected component has one class; validation/test labels must not
    # introduce labels into a supernode containing only held-out nodes.
    nodes = torch.arange(16)
    edges = torch.stack([nodes.repeat(2), torch.cat([nodes.roll(1), nodes.roll(-1)])])
    raw = Data(
        x=graph.x,
        y=torch.arange(32) // 16,
        edge_index=torch.cat([edges, edges + 16], dim=1),
        train_mask=torch.arange(32) % 16 < 12,
        val_mask=torch.arange(32) % 16 == 12,
        test_mask=torch.arange(32) % 16 > 12,
    )
    graph = loader.TransAndInd(splits(raw), "cora", norm=False)
    graph.nclass = 2
    args = quick_args(tmp_path, method, "trans")
    reduced = create_reducer(method, "trans", graph, args).reduce(graph, verbose=False)
    adjacency, features, labels = load_reduced(args)
    assert 0 < len(features) < 32
    assert adjacency.shape == (len(features), len(features))
    assert set(labels.long().tolist()) == {0, 1}
    assert torch.isfinite(features).all() and torch.isfinite(adjacency.to_dense()).all()
    assert reduced is graph


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_multigraph_condensation_saves_batched_graphs_and_trains(tmp_path, monkeypatch, setting):
    from graphslim.dataset.utils import save_reduced

    args = quick_args(tmp_path, "msgc", setting)
    args.condense_model, args.batch_adj = "GCN", 2
    args.outer_loop, args.inner_loop = 1, 1
    args.epochs, args.eval_epochs, args.run_inter_eval = 2, 3, 1
    args.checkpoints = [1]
    sampler = loader.NeighborSampler
    monkeypatch.setattr(
        loader, "NeighborSampler", lambda *a, **kw: sampler(*a, **dict(kw, num_workers=0))
    )
    graph = graph_fixture()
    seed_everything(4)
    reduced = create_reducer("msgc", setting, graph, args).reduce(graph, verbose=False)
    save_reduced(reduced.adj_syn, reduced.feat_syn, reduced.labels_syn, args)
    adjacency, features, labels = load_reduced(args)
    assert adjacency.shape == (2, 12, 12) and features.shape == (12, 8)
    assert labels.shape == (24,) and labels.bincount().tolist() == [12, 12]
    torch.testing.assert_close(adjacency, adjacency.transpose(1, 2))
    assert torch.isfinite(adjacency).all() and (adjacency.diagonal(dim1=1, dim2=2) > 0).all()
    validation, test = Evaluator(args).test(
        reduced, "GCN", reduced=True, mode="cross", verbose=False
    )
    assert 0 <= validation <= 1 and 0 <= test <= 1

"""GDEM eigenbasis condensation on tiny graphs, including disconnected inputs and caches."""

import numpy as np
import pytest
import torch

from graphslim.condensation.gdem import GDEM, MLP
from torch_geometric.data import Data

from graphslim.dataset import load_reduced, loader
from graphslim.dataset.utils import splits
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


def gdem_args(tmp_path, monkeypatch):
    # Run from an empty working directory: nothing may be written relative to it.
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    args = quick_args(tmp_path / "out", "gdem", "trans")
    args.load_path = str(tmp_path / "cache")  # a non-default dataset root
    args.eigen_k, args.e1, args.e2 = 4, 2, 2
    args.epochs, args.eval_epochs, args.run_inter_eval = 12, 2, 1
    args.checkpoints, args._cuda_graph_trace = [11], True
    return args


def disconnected_fixture():
    """A 20-node ring plus a 12-node ring; four training nodes lie outside the LCC."""
    generator = torch.Generator().manual_seed(7)
    nodes = torch.arange(32)

    def ring(members):
        return torch.stack([members.repeat(2), torch.cat([members.roll(1), members.roll(-1)])])

    raw = Data(
        x=torch.rand(32, 8, generator=generator),
        y=nodes % 2,
        edge_index=torch.cat([ring(nodes[:20]), ring(nodes[20:])], dim=1),
        train_mask=(nodes < 14) | ((nodes >= 20) & (nodes < 24)),
        val_mask=(nodes >= 14) & (nodes < 17),
        test_mask=((nodes >= 17) & (nodes < 20)) | (nodes >= 24),
    )
    graph = loader.TransAndInd(splits(raw), "cora", norm=False)
    graph.nclass = 2
    return graph


@pytest.mark.parametrize("fixture", [graph_fixture, disconnected_fixture])
def test_gdem_learns_spectral_graph_and_reuses_preprocessing(tmp_path, monkeypatch, fixture):
    args = gdem_args(tmp_path, monkeypatch)
    graph = fixture()  # torch split indices, as produced by the PyG loaders
    seed_everything(4)
    reducer = create_reducer("gdem", "trans", graph, args)
    n = len(reducer.labels_syn)
    assert n == len(graph.idx_train) // 2 and reducer.eigenvecs_syn.shape == (n, 4)
    initial = reducer.eigenvecs_syn.detach().clone()
    reduced = reducer.reduce(graph, verbose=True)
    assert not torch.equal(initial, reducer.eigenvecs_syn)
    trace = next(r for r in args._cuda_graph_records if r["stage"] == "gdem.spectral_loss")
    losses = trace["loss_trace"]
    assert len(losses) == 12 and np.isfinite(losses).all()
    assert min(losses[1:]) < losses[0]
    cache = tmp_path / "cache" / "cora"
    lcc = np.load(cache / "idx_lcc.npy")
    train_lcc = np.load(cache / "idx_train_lcc.npy")
    idx_map = np.load(cache / "idx_map.npy")
    train = graph.idx_train.tolist()
    # Only training nodes inside the largest component, in split order, with matching map.
    assert train_lcc.tolist() == [i for i in train if i in set(lcc.tolist())]
    assert lcc[idx_map].tolist() == train_lcc.tolist()
    assert len(lcc) == (32 if fixture is graph_fixture else 20)
    assert (cache / "eigenvalues.npy").exists()  # eigen cache lives under args.load_path
    assert list((tmp_path / "work").iterdir()) == []
    adjacency, features, labels = load_reduced(args)
    assert adjacency.shape == (n, n) and features.shape == (n, 8)
    torch.testing.assert_close(adjacency, adjacency.T)
    assert labels.bincount().tolist() == np.bincount(reducer.labels_syn).tolist()
    assert abs(labels.bincount()[0] - labels.bincount()[1]) <= 1
    torch.testing.assert_close(reduced.feat_syn, reducer.x_syn.detach())
    # Second run takes the cached largest-component branch.
    graph = fixture()
    args.epochs, args.checkpoints = 2, [1]
    create_reducer("gdem", "trans", graph, args).reduce(graph, verbose=False)
    assert graph.feat_syn.shape == (n, 8) and torch.isfinite(graph.feat_syn).all()


def test_gdem_feature_initialization_is_deterministic(tmp_path, monkeypatch):
    args = gdem_args(tmp_path, monkeypatch)
    runs = []
    for _ in range(2):
        # Dirty the allocator so uninitialized memory would differ between runs.
        torch.full((12, 8), float("nan")).sum()
        seed_everything(4)
        runs.append(create_reducer("gdem", "trans", graph_fixture(), args).x_syn.detach())
    assert torch.isfinite(runs[0]).all() and runs[0].abs().max() < 100
    torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)


def test_stale_cache_with_non_training_nodes_is_rebuilt(tmp_path, monkeypatch):
    args = gdem_args(tmp_path, monkeypatch)
    cache = tmp_path / "cache" / "cora"
    cache.mkdir(parents=True)
    # A map written by the old refill logic: node 30 is a test node.
    np.save(cache / "idx_lcc.npy", np.arange(32))
    np.save(cache / "idx_train_lcc.npy", np.array([0, 1, 30]))
    np.save(cache / "idx_map.npy", np.array([0, 1, 30]))
    graph = graph_fixture()
    args.epochs, args.checkpoints = 2, [1]
    seed_everything(4)
    create_reducer("gdem", "trans", graph, args).reduce(graph, verbose=False)
    rebuilt = np.load(cache / "idx_train_lcc.npy")
    assert rebuilt.tolist() == graph.idx_train.tolist()


def test_gdem_spectral_helpers_reconstruct_projections(tmp_path, monkeypatch):
    args = gdem_args(tmp_path, monkeypatch)
    agent = object.__new__(GDEM)
    agent.args, agent.device = args, "cpu"
    vectors = torch.eye(3)[:2]  # rows are eigenvectors
    features = torch.arange(6.0).reshape(3, 2)
    per_space = agent.get_eigenspace_embed(vectors, features)
    assert per_space.shape == (2, 3, 2)
    torch.testing.assert_close(per_space[0][0], features[0])
    laplacian = 2 * torch.eye(3)
    real = agent.get_real_embed(2, laplacian, features)
    torch.testing.assert_close(real[1], 4 * features)
    synthetic = agent.get_syn_embed(2, torch.tensor([2.0, 3.0]), vectors, features)
    torch.testing.assert_close(synthetic[1][:2], torch.tensor([[0.0, 4.0], [18.0, 27.0]]))
    eigenvectors = agent.get_init_syn_eigenvecs(7, 3)
    torch.testing.assert_close(eigenvectors.T @ eigenvectors, torch.eye(7), atol=1e-4, rtol=0)
    saved = torch.rand(4, 2)
    (tmp_path / "work" / "initial_feat" / "cora").mkdir(parents=True)
    torch.save(saved, tmp_path / "work" / "initial_feat" / "cora" / f"x_init_0.5_{args.seed}.pt")
    torch.testing.assert_close(agent.get_init_syn_feat("cora", 0.5), saved)


def test_feature_initializer_mlp_uses_dropout_only_in_training():
    torch.manual_seed(0)
    model = MLP(4, 3, 8, dropout=0.5)
    assert model.layers[0].bias.abs().sum() == 0
    features = torch.rand(5, 4)
    model.eval()
    first, second = model(features), model(features)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.exp().sum(1), torch.ones(5))
    model.train()
    assert not torch.equal(model(features), first)

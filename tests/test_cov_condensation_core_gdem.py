"""GDEM eigenbasis condensation on tiny graphs, including disconnected inputs and caches."""

import numpy as np
import pytest
import torch

from graphslim.condensation.gdem import GDEM, MLP
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


def gdem_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cora").mkdir(parents=True)
    args = quick_args(tmp_path / "out", "gdem", "trans")
    args.load_path = "./data"  # GDEM's eigen cache is written relative to the working directory
    args.eigen_k, args.e1, args.e2 = 4, 2, 2
    args.epochs, args.eval_epochs, args.run_inter_eval = 12, 2, 1
    args.checkpoints, args._cuda_graph_trace = [11], True
    return args


def numpy_split_graph():
    # GraphSAINT-format loaders provide NumPy split indices; get_train_lcc requires them
    # (torch index tensors are hashed by identity there, see the coverage report).
    graph = graph_fixture()
    graph.idx_train = graph.idx_train.numpy()
    return graph


def xavier_feature_init(self):
    """The documented Xavier start; the package skips it (see coverage report) and
    optimizes uninitialized ``torch.empty`` memory, which makes runs nondeterministic.
    ``init_feat`` itself is covered in test_gdem_feature_initialization.py."""
    torch.nn.init.xavier_uniform_(self.x_syn.data)
    return self.x_syn


def test_gdem_learns_spectral_graph_and_reuses_preprocessing(tmp_path, monkeypatch):
    args = gdem_args(tmp_path, monkeypatch)
    monkeypatch.setattr(GDEM, "init_feat", xavier_feature_init)
    graph = numpy_split_graph()
    seed_everything(4)
    reducer = create_reducer("gdem", "trans", graph, args)
    assert reducer.eigenvecs_syn.shape == (12, 4)
    initial = reducer.eigenvecs_syn.detach().clone()
    reduced = reducer.reduce(graph, verbose=True)
    assert not torch.equal(initial, reducer.eigenvecs_syn)
    trace = next(r for r in args._cuda_graph_records if r["stage"] == "gdem.spectral_loss")
    losses = trace["loss_trace"]
    assert len(losses) == 12 and np.isfinite(losses).all()
    assert min(losses[1:]) < losses[0]
    lcc = np.load(tmp_path / "data" / "cora" / "idx_lcc.npy")
    assert sorted(lcc) == list(range(32))
    train_lcc = np.load(tmp_path / "data" / "cora" / "idx_train_lcc.npy")
    assert sorted(train_lcc) == sorted(graph.idx_train)
    assert (tmp_path / "data" / "cora" / "eigenvalues.npy").exists()
    adjacency, features, labels = load_reduced(args)
    assert adjacency.shape == (12, 12) and features.shape == (12, 8)
    torch.testing.assert_close(adjacency, adjacency.T)
    assert labels.bincount().tolist() == [6, 6]
    torch.testing.assert_close(reduced.feat_syn, reducer.x_syn.detach())
    # Second run takes the cached largest-component branch.
    graph = numpy_split_graph()
    args.epochs, args.checkpoints = 2, [1]
    create_reducer("gdem", "trans", graph, args).reduce(graph, verbose=False)
    assert graph.feat_syn.shape == (12, 8)


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
    (tmp_path / "initial_feat" / "cora").mkdir(parents=True)
    torch.save(saved, tmp_path / "initial_feat" / "cora" / f"x_init_0.5_{args.seed}.pt")
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

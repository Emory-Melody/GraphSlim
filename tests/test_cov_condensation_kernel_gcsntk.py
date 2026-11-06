"""GCSNTK kernel-ridge condensation and the Mirage logging helper."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import graphslim.condensation.gcsntk as gcsntk_module
from graphslim.condensation.gcsntk import GCSNTK
from graphslim.condensation.mirage import Mirage
from graphslim.dataset import load_reduced
from graphslim.models import KernelRidgeRegression, StructureBasedNeuralTangentKernel
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gcsntk_args(tmp_path, dataset="cora"):
    args = quick_args(tmp_path, "gcsntk", "trans")
    args.dataset = dataset
    args.K, args.L, args.k, args.ridge, args.scale, args.adj = 1, 1, 2, 1e-3, "add", False
    args.lr, args.epochs, args.eval_epochs, args.run_inter_eval = 0.01, 4, 3, 1
    args.checkpoints = [1, 3]
    args.accumulate_steps, args.batch_size = None, 8
    return args


def binary_sparse(dense):
    index = dense.nonzero().T
    return torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), tuple(dense.shape))


@pytest.mark.parametrize("structure", [False, True])
def test_gcsntk_reduces_and_saves_soft_label_graph(tmp_path, structure):
    args = gcsntk_args(tmp_path)
    args.adj = structure
    graph = tiny_graph()
    seed_everything(3)
    reducer = create_reducer("gcsntk", "trans", graph, args)
    assert (reducer.K, reducer.L, reducer.k, reducer.ridge) == (1, 1, 2, 1e-3)
    losses = []
    evaluate = reducer.intermediate_evaluation

    def record(best_val, loss_avg=None, **kw):
        losses.append(loss_avg)
        return evaluate(best_val, loss_avg, **kw)

    reducer.intermediate_evaluation = record
    reduced = reducer.reduce(graph, verbose=False)
    size = round(len(graph.idx_train) * args.reduction_rate)
    assert reduced.feat_syn.shape == (size, 8)
    assert reduced.labels_syn.shape == (size, 2)  # learned soft targets
    assert reduced.adj_syn.shape == (size, size)
    assert len(losses) == 2 and all(np.isfinite(losses))
    if structure:
        # update_E links each node to itself and its nearest neighbours: 8 + 4*8/2 edges.
        assert reduced.adj_syn.sum().item() == size + 2 * size
        torch.testing.assert_close(reduced.adj_syn, reduced.adj_syn.T)
    else:
        torch.testing.assert_close(reduced.adj_syn, torch.eye(size))
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (size, 8) and labels.shape == (size, 2)
    assert torch.isfinite(adjacency).all() and torch.isfinite(features).all()


def test_gcsntk_training_step_reduces_kernel_regression_loss(tmp_path):
    args = gcsntk_args(tmp_path)
    graph = tiny_graph()
    reducer = GCSNTK("trans", graph, args)
    torch.manual_seed(0)
    x_s = torch.rand(6, 8, requires_grad=True)
    y_s = torch.rand(6, 2, requires_grad=True)
    optimizer = torch.optim.Adam([x_s, y_s], lr=0.05)
    kernel = StructureBasedNeuralTangentKernel(K=1, L=1, scale="add")
    krr = KernelRidgeRegression(kernel.nodes_gram, torch.tensor(1e-3))
    target = F.one_hot(graph.labels_train, 2)
    adjacency = torch.as_tensor(graph.adj_train.toarray()) + torch.eye(16)
    losses = []
    for _ in range(15):
        returned_x, returned_y, loss, accuracy = reducer.train(
            krr,
            graph.feat_train,
            x_s,
            target,
            y_s,
            binary_sparse(adjacency),
            binary_sparse(torch.eye(6)),
            torch.nn.MSELoss(),
            optimizer,
        )
        assert returned_x is x_s and returned_y is y_s
        assert 0 <= accuracy <= 100
        losses.append(loss)
    assert losses[-1] < losses[0]
    loss, accuracy = reducer.test(
        krr,
        graph.feat_train,
        x_s,
        target,
        y_s,
        binary_sparse(adjacency),
        binary_sparse(torch.eye(6)),
        torch.nn.MSELoss(),
    )
    assert loss == pytest.approx(losses[-1], rel=0.2)
    assert 0 <= accuracy <= 100


def test_gcsntk_accumulates_gradients_between_optimizer_steps(tmp_path):
    args = gcsntk_args(tmp_path)
    reducer = GCSNTK("trans", tiny_graph(), args)
    x_s = torch.rand(4, 3, requires_grad=True)
    y_s = torch.rand(4, 2, requires_grad=True)
    target = F.one_hot(torch.tensor([0, 1, 0, 1, 1]), 2)
    query = torch.rand(5, 3)
    krr = KernelRidgeRegression(lambda a, b, *_: a @ b.T, 1e-2)
    steps = []

    class Recorder(torch.optim.SGD):
        def step(self, closure=None):
            steps.append(x_s.grad.clone())
            return super().step(closure)

    optimizer = Recorder([x_s, y_s], lr=0.1)
    for i in range(3):
        reducer.train(
            krr, query, x_s, target, y_s, None, None, torch.nn.MSELoss(), optimizer, 2, i, 3
        )
    # Steps happen at the accumulation boundary (i=1) and for the final batch (i=2).
    assert len(steps) == 2
    assert x_s.grad is None or torch.count_nonzero(x_s.grad) == 0


class SyntheticBatches:
    """Partitioned training batches with the ``LargeDataLoader`` interface."""

    def __init__(self, graph, batches=2):
        self.graph = graph
        self.batches = np.array_split(np.arange(len(graph.idx_train)), batches)
        self.split = 0

    def properties(self):
        g = self.graph
        return len(self.batches), len(g.idx_train), g.nclass, g.feat_full.shape[1], g.num_nodes

    def split_batch(self):
        self.split += 1

    def get_batch(self, i):
        index = self.batches[i]
        adjacency = torch.as_tensor(self.graph.adj_train[index][:, index].toarray())
        adjacency = adjacency + torch.eye(len(index))
        return (
            self.graph.feat_train[index],
            self.graph.labels_train[index].reshape(-1, 1),
            binary_sparse(adjacency),
        )


@pytest.mark.parametrize("structure,accumulate", [(False, None), (True, 2)])
def test_gcsntk_large_dataset_branch_trains_over_partitions(
    tmp_path, monkeypatch, structure, accumulate
):
    args = gcsntk_args(tmp_path, dataset="flickr")
    args.adj, args.accumulate_steps = structure, accumulate
    graph = tiny_graph()
    created = []

    def loader(name, split, batch_size, split_method):
        assert (name, split, batch_size, split_method) == ("flickr", "train", 8, "kmeans")
        created.append(SyntheticBatches(graph, batches=3))
        return created[-1]

    monkeypatch.setattr(gcsntk_module, "LargeDataLoader", loader)
    seed_everything(5)
    reducer = GCSNTK("ind", graph, args)
    reduced = reducer.reduce(graph, verbose=False)
    assert created[0].split == 1
    size = round(len(graph.idx_train) * args.reduction_rate)
    assert reduced.feat_syn.shape == (size, 8) and reduced.labels_syn.shape == (size, 2)
    assert torch.isfinite(reduced.feat_syn).all()
    if not structure:
        torch.testing.assert_close(reduced.adj_syn, torch.eye(size))
    adjacency, features, _ = load_reduced(args)
    assert features.shape == (size, 8) and adjacency.shape == (size, size)


def test_mirage_appends_timestamped_messages(tmp_path):
    args = quick_args(tmp_path, "gcond", "trans")
    graph = tiny_graph()
    agent = Mirage("trans", graph, args)
    assert agent.feat_syn.shape == (len(graph.labels_syn), 8)
    log = tmp_path / "mirage.log"
    agent.append_to_file("first entry", log)
    agent.append_to_file({"seed": 7, "hops": [1, 2]}, log)
    text = log.read_text(encoding="utf-8")
    blocks = text.split("-" * 20 + "\n")[1:]
    assert len(blocks) == 2
    assert blocks[0].splitlines()[1] == "first entry"
    assert "{'hops': [1, 2], 'seed': 7}" in blocks[1]
    for block in blocks:
        # day-month-year hour:minute:second
        stamp = block.splitlines()[0]
        assert len(stamp) == 19 and stamp[2] == stamp[5] == "-" and stamp[13] == ":"

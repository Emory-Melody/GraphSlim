# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""GEOM and SFGC: expert buffer creation, buffer loading, and trajectory matching."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from graphslim.condensation.geom import GEOM
from graphslim.condensation.sfgc import SFGC
from graphslim.dataset import load_reduced
from graphslim.models import GCN
from graphslim.reduction import create_reducer
from graphslim.utils import normalize_adj_tensor, seed_everything, to_tensor
from test_legacy_reducer_integration import graph_fixture
from test_lightweight_methods import quick_args


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    # Both reducers write buffers to ``../<method>_buffer`` relative to cwd.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def trajectory_args(tmp_path, method, setting, optim="Adam"):
    args = quick_args(tmp_path / "checkpoints", method, setting)
    args.dataset = "cora"
    args.hidden, args.init, args.optim = 8, "random", optim
    args.no_buff, args.teacher_epochs, args.expert_epochs = True, 11, 10
    args.epochs, args.syn_steps, args.start_epoch = 2, 2, 0
    args.eval_epochs, args.run_inter_eval = 3, 1
    args.checkpoints = [1]
    args.lr_teacher, args.lr_student, args.wd_teacher, args.mom_teacher = 0.01, 0.01, 0, 0
    args.lr_feat = 0.01
    args.min_start_epoch, args.max_start_epoch_s, args.max_start_epoch = 0, 1, 1
    args.lam, args.T, args.scheduler = 0.5, 5, "linear"
    args.soft_label, args.optim_lr, args.beta = False, True, 0
    args.lr_y, args.lr_tem, args.tem, args.maxtem = 0.001, 0.001, 1.0, 2.0
    args.num_experts = 10
    return args


def expert_trajectory(graph, args, setting, epochs, seed):
    """Train one GCN expert and keep cloned (not aliased) parameter snapshots."""
    torch.manual_seed(seed)
    if setting == "trans":
        feat, adj, labels = to_tensor(graph.feat_full, graph.adj_full, label=graph.labels_full)
        index = torch.as_tensor(graph.idx_train)
    else:
        feat, adj, labels = to_tensor(graph.feat_train, graph.adj_train, label=graph.labels_train)
        index = torch.arange(len(labels))
    adj = normalize_adj_tensor(adj, sparse=True)
    model = GCN(feat.shape[1], args.hidden, graph.nclass, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    snapshots = [[p.detach().clone() for p in model.parameters()]]
    for _ in range(epochs):
        optimizer.zero_grad()
        F.nll_loss(model(feat, adj)[index], labels[index]).backward()
        optimizer.step()
        snapshots.append([p.detach().clone() for p in model.parameters()])
    return snapshots


def write_buffers(directory, graph, args, setting, files=2, per_file=2, epochs=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed = 0
    for n in range(files):
        trajectories = []
        for _ in range(per_file):
            trajectories.append(expert_trajectory(graph, args, setting, epochs, seed))
            seed += 1
        torch.save(trajectories, directory / f"replay_buffer_{n}.pt")


def gcn_shapes(graph, args):
    model = GCN(graph.feat_full.shape[1], args.hidden, graph.nclass, args)
    return [tuple(p.shape) for p in model.parameters()]


# --------------------------------------------------------------------------- GEOM


@pytest.mark.parametrize(
    "setting,optim,scheduler", [("trans", "Adam", "linear"), ("ind", "SGD", "root")]
)
def test_geom_curriculum_experts_are_saved_as_replay_buffers(
    tmp_path, workdir, setting, optim, scheduler
):
    args = trajectory_args(tmp_path, "geom", setting, optim)
    args.scheduler, args.teacher_epochs = scheduler, 21
    graph = graph_fixture()
    seed_everything(3)
    reducer = GEOM(setting, graph, args)
    assert Path(reducer.buf_dir).resolve().parent == (tmp_path / "geom_buffer").resolve()
    assert args.condense_model == "GCN"
    reducer.buffer_cl(graph)
    reducer.buffer_cl(graph)  # a second run appends replay_buffer_1 instead of overwriting
    files = sorted(Path(reducer.buf_dir).glob("replay_buffer_*.pt"))
    assert [f.name for f in files] == ["replay_buffer_0.pt", "replay_buffer_1.pt"]
    trajectories = torch.load(files[0], weights_only=True)
    assert len(trajectories) == args.num_experts
    shapes = gcn_shapes(graph, args)
    for trajectory in trajectories:
        # initial snapshot plus epochs 10 and 20 of the 22 teacher epochs
        assert len(trajectory) == 3
        for snapshot in trajectory:
            assert [tuple(p.shape) for p in snapshot] == shapes
            assert all(torch.isfinite(p).all() for p in snapshot)
    # The loaded buffer matches what was written to the first discovered file.
    file_idx, expert_idx, expert_files = reducer.expert_load()
    assert (file_idx, expert_idx) == (0, 0)
    assert sorted(Path(f).name for f in expert_files) == [f.name for f in files]
    assert len(reducer.buffer) == args.num_experts


def test_geom_expert_load_reports_missing_buffers(tmp_path, workdir):
    args = trajectory_args(tmp_path, "geom", "trans")
    reducer = GEOM("trans", graph_fixture(), args)
    assert Path(reducer.buf_dir).is_dir()
    # Re-creating the reducer reuses the existing (empty) buffer directory.
    reducer = GEOM("trans", graph_fixture(), args)
    with pytest.raises(AssertionError, match="No buffers detected"):
        reducer.expert_load()


@pytest.mark.parametrize(
    "setting,soft_label,beta,optim_lr,optim",
    [
        ("trans", False, 0, True, "Adam"),
        ("trans", False, 0.1, False, "Adam"),
        ("trans", True, 0.1, True, "Adam"),
        ("ind", True, 0.1, True, "Adam"),
        ("ind", False, 0, True, "SGD"),
    ],
)
def test_geom_matches_expert_trajectories_and_saves_reloadable_graph(
    tmp_path, workdir, setting, soft_label, beta, optim_lr, optim
):
    args = trajectory_args(tmp_path, "geom", setting, optim)
    args.soft_label, args.beta, args.optim_lr = soft_label, beta, optim_lr
    args.tem = 3.0
    args.epochs, args.checkpoints = 5, [0, 4]
    graph = graph_fixture()
    seed_everything(5)
    reducer = create_reducer("geom", setting, graph, args)
    write_buffers(reducer.buf_dir, graph, args, setting)
    losses = []
    evaluate = reducer.intermediate_evaluation

    def record(best_val, loss_avg=None, **kw):
        losses.append(loss_avg)
        return evaluate(best_val, loss_avg, **kw)

    reducer.intermediate_evaluation = record
    initial = None

    def init(with_adj=False, reuse_init=False):
        nonlocal initial
        feat, adj = type(reducer).init(reducer, with_adj=with_adj)
        initial = torch.as_tensor(feat).clone()
        return feat, adj

    reducer.init = init
    reduced = reducer.reduce(graph, verbose=False)
    assert len(losses) == 2 and all(np.isfinite(losses))
    assert torch.isfinite(reduced.feat_syn).all()
    assert not torch.equal(reduced.feat_syn, initial)  # features were optimized
    torch.testing.assert_close(reduced.adj_syn, torch.eye(12))
    if soft_label:
        # Soft labels are learnable per-class log-probability rows.
        assert reduced.labels_syn.shape == (12, 2)
        assert torch.isfinite(reduced.labels_syn).all()
        if setting == "ind":
            # The temperature starts above ``maxtem`` and is clipped to it.
            assert reducer.tem.item() == args.maxtem
    else:
        assert torch.equal(reduced.labels_syn, torch.as_tensor(graph.labels_syn))
    if optim_lr:
        assert reducer.syn_lr.requires_grad
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (12, 8) and adjacency.shape == (12, 12)
    assert labels.shape[0] == 12


def test_geom_matching_loss_decreases_for_a_fixed_expert(tmp_path, workdir):
    args = trajectory_args(tmp_path, "geom", "trans")
    args.epochs, args.syn_steps, args.lr_feat, args.optim_lr = 12, 3, 0.05, False
    args.checkpoints = list(range(args.epochs))
    args.run_inter_eval, args.eval_epochs = 1, 1
    graph = graph_fixture()
    seed_everything(9)
    reducer = GEOM("trans", graph, args)
    write_buffers(reducer.buf_dir, graph, args, "trans", files=1, per_file=1)
    losses = []
    reducer.intermediate_evaluation = lambda best, loss=None, **kw: losses.append(loss) or best
    reducer.reduce(graph, verbose=False)
    assert len(losses) == args.epochs
    assert losses[-1] < losses[0]


def test_geom_reddit_first_iteration_uses_initial_sparse_structure(tmp_path, workdir):
    args = trajectory_args(tmp_path, "geom", "trans")
    args.dataset, args.reduction_rate = "reddit", 0.05
    args.epochs, args.checkpoints = 2, [1]
    graph = graph_fixture()
    seed_everything(2)
    reducer = GEOM("trans", graph, args)
    assert "reddit" in reducer.buf_dir
    write_buffers(reducer.buf_dir, graph, args, "trans", files=1, per_file=2)
    reduced = reducer.reduce(graph, verbose=False)
    assert torch.isfinite(reduced.feat_syn).all()
    assert reduced.feat_syn.shape[0] == len(graph.labels_syn)


class _StopAfterBuffers(Exception):
    pass


def _stop(**kwargs):
    raise _StopAfterBuffers


def test_geom_reduce_builds_twenty_experts_before_matching(tmp_path, workdir):
    args = trajectory_args(tmp_path, "geom", "trans")
    args.no_buff, args.teacher_epochs = False, 2
    graph = graph_fixture()
    seed_everything(1)
    reducer = GEOM("trans", graph, args)
    reducer.init = _stop
    with pytest.raises(_StopAfterBuffers):
        reducer.reduce(graph, verbose=False)
    assert args.num_experts == 20  # reduce() fixes the expert count
    files = sorted(Path(reducer.buf_dir).glob("replay_buffer_*.pt"))
    assert len(files) == 2
    assert all(len(torch.load(f, weights_only=True)) == 10 for f in files)


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_geom_coreset_selection_round_trip(tmp_path, workdir, setting):
    args = trajectory_args(tmp_path, "geom", setting)
    args.lr_coreset, args.coreset_epochs = 0.05, 3
    graph = graph_fixture()
    seed_everything(6)
    reducer = GEOM(setting, graph, args)
    selected = reducer.init_coreset_select(graph)
    saved = Path(reducer.buf_dir) / f"idx_cora_{args.reduction_rate}_kcenter_{args.seed}.npy"
    assert saved.is_file()
    np.testing.assert_array_equal(np.load(saved), selected)
    assert len(np.unique(selected)) == len(selected) == 12
    feat, adj, labels = to_tensor(graph.feat_full, graph.adj_full, label=graph.labels_full)
    feat_s, adj_s, labels_s = reducer.get_coreset_init(feat, adj.to_dense(), labels)
    np.testing.assert_allclose(feat_s, feat.numpy()[selected])
    assert tuple(adj_s.shape) == (12, 12)
    assert torch.equal(labels_s, labels[selected])


def test_geom_synset_save_returns_detached_copies(tmp_path, workdir):
    args = trajectory_args(tmp_path, "geom", "trans")
    graph = graph_fixture()
    reducer = GEOM("trans", graph, args)
    reducer.feat_syn.data.copy_(torch.randn(reducer.feat_syn.shape))
    reducer.labels_syn = torch.as_tensor(graph.labels_syn)
    feat, adj, labels = reducer.synset_save()
    assert not feat.requires_grad
    assert feat.data_ptr() != reducer.feat_syn.data_ptr()
    torch.testing.assert_close(feat, reducer.feat_syn.detach())
    torch.testing.assert_close(adj, torch.eye(len(feat)))
    assert torch.equal(labels, reducer.labels_syn)


# --------------------------------------------------------------------------- SFGC


@pytest.mark.parametrize("setting,optim", [("trans", "Adam"), ("ind", "SGD")])
def test_sfgc_expert_stage_writes_two_buffer_files(tmp_path, workdir, setting, optim):
    args = trajectory_args(tmp_path, "sfgc", setting, optim)
    args.no_buff, args.teacher_epochs = False, 12
    graph = graph_fixture()
    seed_everything(4)
    reducer = SFGC(setting, graph, args)
    assert Path(reducer.buf_dir).resolve().parent == (tmp_path / "sfgc_buffer").resolve()
    reducer.init = _stop
    with pytest.raises(_StopAfterBuffers):
        reducer.reduce(graph, verbose=False)
    files = sorted(Path(reducer.buf_dir).glob("replay_buffer_*.pt"))
    assert [f.name for f in files] == ["replay_buffer_0.pt", "replay_buffer_1.pt"]
    shapes = gcn_shapes(graph, args)
    for path in files:
        trajectories = torch.load(path, weights_only=True)
        assert len(trajectories) == 10
        for trajectory in trajectories:
            # initial snapshot plus epoch 10 of 12 teacher epochs
            assert len(trajectory) == 2
            assert [tuple(p.shape) for p in trajectory[0]] == shapes
            assert all(torch.isfinite(p).all() for s in trajectory for p in s)


def test_sfgc_expert_load_reports_missing_buffers(tmp_path, workdir):
    args = trajectory_args(tmp_path, "sfgc", "trans")
    reducer = SFGC("trans", graph_fixture(), args)
    reducer = SFGC("trans", graph_fixture(), args)  # existing directory is reused
    with pytest.raises(AssertionError, match="No buffers detected"):
        reducer.expert_load(reducer.buf_dir)


@pytest.mark.parametrize("setting,optim", [("trans", "Adam"), ("ind", "SGD")])
def test_sfgc_matches_trajectories_rolls_over_buffers_and_saves(
    tmp_path, workdir, monkeypatch, setting, optim
):
    args = trajectory_args(tmp_path, "sfgc", setting, optim)
    args.start_epoch, args.expert_epochs = 1 if optim == "SGD" else 10, 20
    args.epochs, args.checkpoints = 6, [0, 5]
    graph = graph_fixture()
    seed_everything(8)
    reducer = create_reducer("sfgc", setting, graph, args)
    write_buffers(reducer.buf_dir, graph, args, setting, files=2, per_file=2, epochs=4)
    loads = []
    original_load = torch.load

    def counting_load(*a, **kw):
        loads.append(a[0])
        return original_load(*a, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", counting_load)
        reduced = reducer.reduce(graph, verbose=False)
    # One initial load plus a reload after every two consumed experts.
    assert len(loads) == 1 + args.epochs // 2
    assert {Path(p).name for p in loads} == {"replay_buffer_0.pt", "replay_buffer_1.pt"}
    assert torch.isfinite(reduced.feat_syn).all()
    torch.testing.assert_close(reduced.adj_syn, torch.eye(12))
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (12, 8) and adjacency.shape == (12, 12)
    assert torch.equal(labels, torch.as_tensor(graph.labels_syn))


def nan_buffer(graph, args, directory):
    trajectory = expert_trajectory(graph, args, "trans", epochs=3, seed=0)
    trajectory[-1] = [torch.full_like(p, float("nan")) for p in trajectory[-1]]
    trajectory[1] = [torch.full_like(p, float("nan")) for p in trajectory[1]]
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.save([trajectory], Path(directory) / "replay_buffer_0.pt")


@pytest.mark.parametrize("method", ["geom", "sfgc"])
def test_matching_stops_before_checkpoints_when_loss_is_nan(tmp_path, workdir, method):
    args = trajectory_args(tmp_path, method, "trans")
    args.epochs, args.checkpoints, args.expert_epochs = 4, [0, 1, 2, 3], 10
    graph = graph_fixture()
    seed_everything(0)
    reducer = create_reducer(method, "trans", graph, args)
    nan_buffer(graph, args, reducer.buf_dir)
    calls = []
    reducer.intermediate_evaluation = lambda *a, **kw: calls.append(a) or 0
    reducer.reduce(graph, verbose=False)
    assert calls == []
    with pytest.raises(FileNotFoundError):
        load_reduced(args)


def test_geom_rejects_unsupported_expert_optimizer(tmp_path, workdir):
    args = trajectory_args(tmp_path, "geom", "trans", optim="RMSprop")
    graph = graph_fixture()
    reducer = GEOM("trans", graph, args)
    with pytest.raises(ValueError, match="'Adam' or 'SGD', got 'RMSprop'"):
        reducer.buffer_cl(graph)
    assert not list(Path(reducer.buf_dir).glob("replay_buffer_*.pt"))

"""Train actual experts and use their saved trajectories for graph condensation."""

from pathlib import Path

import pytest
import torch

from graphslim.dataset import load_reduced
from graphslim.dataset.utils import save_reduced
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


def trajectory_args(tmp_path, method, setting):
    args = quick_args(tmp_path / "checkpoints", method, setting)
    args.hidden, args.init, args.optim = 8, "random", "Adam"
    args.no_buff, args.teacher_epochs, args.expert_epochs = False, 11, 10
    args.epochs, args.syn_steps, args.start_epoch = 2, 2, 0
    args.eval_epochs, args.run_inter_eval = 3, 1
    args.checkpoints = [1]
    args.lr_teacher, args.lr_student, args.wd_teacher = 0.01, 0.01, 0
    args.min_start_epoch, args.max_start_epoch_s, args.max_start_epoch = 0, 1, 1
    args.lam, args.T, args.scheduler = 0.5, 5, "linear"
    args.soft_label, args.optim_lr, args.beta = False, True, 0
    args.lr_y, args.lr_tem, args.tem, args.maxtem = 0.001, 0.001, 1.0, 2.0
    return args


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize(
    "method,soft_label,beta",
    [("sfgc", False, 0), ("geom", False, 0), ("geom", False, 0.1), ("geom", True, 0.1)],
)
def test_real_experts_have_independent_timestamps_and_produce_reloadable_graphs(
    tmp_path, monkeypatch, method, soft_label, beta, setting
):
    # Legacy reducers place buffers beside cwd. Keep the real on-disk workflow
    # inside the test directory, without replacing buffer loading or training.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    args = trajectory_args(tmp_path, method, setting)
    args.soft_label, args.beta = soft_label, beta
    graph = graph_fixture()
    seed_everything(17)
    reducer = create_reducer(method, setting, graph, args)
    reduced = reducer.reduce(graph, verbose=False)
    files = sorted(Path(reducer.buf_dir).glob("replay_buffer_*.pt"))
    assert len(files) == 2
    trajectories = torch.load(files[0], weights_only=True)
    assert len(trajectories) == 10
    for trajectory in trajectories:
        # Regression: on CPU, ``p.detach().cpu()`` aliased the live weights, so every
        # snapshot equalled the final state and the matching loss divided by zero.
        initial, trained = trajectory[0], trajectory[-1]
        assert len(initial) == len(trained)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(initial, trained))
        assert any(not torch.equal(a, b) for a, b in zip(initial, trained))
        assert all(torch.isfinite(value).all() for value in trained)
    assert torch.isfinite(reduced.feat_syn).all()
    torch.testing.assert_close(reduced.adj_syn, torch.eye(12))
    save_reduced(reduced.adj_syn, reduced.feat_syn, reduced.labels_syn, args)
    adjacency, features, labels = load_reduced(args)
    assert features.shape == (12, 8) and adjacency.shape == (12, 12)
    assert labels.shape == ((12, 2) if soft_label else (12,))
    assert torch.isfinite(labels).all()
    if soft_label and setting == "ind":
        # Regression: the temperature optimizer used to call zero_grad() instead of step().
        assert reducer.tem.item() != args.tem

    # Resume the real stored experts and exercise buffer-file rollover.
    args.no_buff, args.epochs, args.checkpoints = True, 12, [11]
    fresh = graph_fixture()
    cached = create_reducer(method, setting, fresh, args).reduce(fresh, verbose=False)
    assert torch.isfinite(cached.feat_syn).all()
    assert len(list(Path(reducer.buf_dir).glob("replay_buffer_*.pt"))) == 2


@pytest.mark.parametrize("method", ["geom", "sfgc"])
def test_missing_expert_buffers_are_reported(tmp_path, monkeypatch, method):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    args = trajectory_args(tmp_path, method, "trans")
    reducer = create_reducer(method, "trans", graph_fixture(), args)
    with pytest.raises(AssertionError, match="No buffers detected"):
        if method == "geom":
            reducer.expert_load()
        else:
            reducer.expert_load(reducer.buf_dir)

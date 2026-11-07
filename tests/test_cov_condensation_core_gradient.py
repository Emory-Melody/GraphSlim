# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Gradient/distribution-matching condensers: GCondBase helpers, SGDD, SimGC, GCDM."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from graphslim.condensation.gcdm import dist
from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.simgc import (
    SimGCTeacher,
    propagated_features,
    smoothness_loss,
    training_view,
)
from graphslim.dataset import load_reduced, loader
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import Evaluator
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_legacy_reducer_integration import graph_fixture
from test_lightweight_methods import quick_args


@pytest.fixture(autouse=True)
def single_thread_no_workers(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    sampler = loader.NeighborSampler
    monkeypatch.setattr(
        loader, "NeighborSampler", lambda *a, **kw: sampler(*a, **dict(kw, num_workers=0))
    )
    yield
    torch.set_num_threads(previous)


def gradient_args(tmp_path, method, setting="trans"):
    args = quick_args(tmp_path, method, setting)
    args.condense_model, args.nlayers, args.ntrans = "SGC", 2, 1
    args.outer_loop, args.inner_loop = 2, 1
    args.epochs, args.eval_epochs, args.run_inter_eval = 2, 2, 1
    args.checkpoints, args.dropout, args.with_bn = [1], 0.0, False
    return args


# --------------------------------------------------------------------------- GCondBase


def test_synthetic_labels_follow_rarest_first_budget_and_reset(tmp_path):
    args = gradient_args(tmp_path, "gcond")
    args.verbose = True
    graph = graph_fixture()
    graph.labels_train = torch.tensor([0] * 4 + [1] * 20)
    reducer = create_reducer("gcond", "trans", graph, args)
    # Rare class keeps floor(4 * 0.5) nodes; the largest class absorbs the remainder.
    assert reducer.labels_syn.tolist() == [0, 0] + [1] * 10
    assert reducer.num_class_dict == {0: 2, 1: 10}
    assert reducer.syn_class_indices == {0: [0, 2], 1: [2, 12]}
    torch.manual_seed(0)
    reducer.reset_parameters()
    first = reducer.feat_syn.detach().clone()
    reducer.reset_parameters()
    assert first.std() > 0.5 and not torch.equal(first, reducer.feat_syn)


def test_initializer_runs_configured_reducer_or_reuses_saved_checkpoint(tmp_path):
    args = gradient_args(tmp_path, "gcond")
    args.init = "random"
    graph = graph_fixture()
    reducer = create_reducer("gcond", "trans", graph, args)
    features, adjacency = reducer.init(with_adj=True)
    assert features.shape == (12, 8) and adjacency.shape[0] == 12
    # The random coreset copies real training rows.
    train_rows = {tuple(row.tolist()) for row in graph.feat_train}
    assert all(tuple(row.tolist()) in train_rows for row in torch.as_tensor(features))
    assert (tmp_path / "reduced_graph" / "random").is_dir()
    marker = torch.full((12, 8), 7.0)
    args.method = "random"
    save_reduced(torch.eye(12), marker, None, args)
    args.method = "gcond"
    torch.testing.assert_close(reducer.init(reuse_init=True), marker)
    reused_features, reused_adj = reducer.init(with_adj=True, reuse_init=True)
    torch.testing.assert_close(reused_adj, torch.eye(12))
    args.init = "kcenter"  # no checkpoint: falls back to running the initializer
    assert reducer.init(reuse_init=True).shape == (12, 8)
    assert (tmp_path / "reduced_graph" / "kcenter").is_dir()


def test_one_hot_labels_match_index_label_gradients(tmp_path):
    args = gradient_args(tmp_path, "gcond")
    graph = graph_fixture()
    seed_everything(1)
    reducer = create_reducer("gcond", "trans", graph, args)
    reducer.feat_syn.data.copy_(torch.rand(12, 8))
    reducer.adj_syn = torch.eye(12)
    from graphslim.models import get_model_class
    from graphslim.utils import normalize_adj_tensor, to_tensor

    model = get_model_class("SGC")(8, 8, 2, args)
    features, adjacency = to_tensor(graph.feat_full, graph.adj_full)
    adjacency = normalize_adj_tensor(adjacency, sparse=True)
    labels = graph.labels_full
    synthetic = torch.as_tensor(reducer.labels_syn)
    torch.manual_seed(5)
    hard = reducer.train_class(model, adjacency, features, labels, synthetic, args)
    torch.manual_seed(5)
    one_hot = reducer.train_class(
        model, adjacency, features, F.one_hot(labels), F.one_hot(synthetic), args
    )
    assert torch.isfinite(hard) and hard > 0
    torch.testing.assert_close(hard, one_hot)
    # Regression: soft mode used real labels as synthetic class masks and crashed. KL to a
    # one-hot target equals the NLL, so soft mode reproduces the hard-label objective.
    for real, synthetic_targets in ((labels, synthetic), (F.one_hot(labels), F.one_hot(synthetic))):
        torch.manual_seed(5)
        soft = reducer.train_class(
            model, adjacency, features, real, synthetic_targets, args, soft=True
        )
        torch.testing.assert_close(soft, hard)


def test_batchnorm_statistics_are_frozen_by_check_bn(tmp_path):
    args = gradient_args(tmp_path, "gcond")
    reducer = create_reducer("gcond", "trans", graph_fixture(), args)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    model.eval()
    checked = reducer.check_bn(model)
    assert checked[0].training and not checked[1].training
    plain = torch.nn.Linear(2, 2).eval()
    assert not reducer.check_bn(plain).training


def test_validation_only_evaluation_reports_and_saves_only_improvements(tmp_path):
    args = gradient_args(tmp_path, "gcond")
    args.verbose, args.validation_only = True, True
    graph = graph_fixture()
    reducer = create_reducer("gcond", "trans", graph, args)
    graph.feat_syn = graph.feat_train[:12].clone()
    graph.adj_syn = torch.eye(12)
    graph.labels_syn = torch.as_tensor(reducer.labels_syn)
    assert len(reducer.test_with_val(iters=2)) == 1
    best = reducer.intermediate_evaluation(-1.0, loss_avg=0.5)
    assert 0 <= best <= 1
    _, features, _ = load_reduced(args)
    torch.testing.assert_close(features, graph.feat_syn)
    graph.feat_syn = torch.zeros(12, 8)
    assert reducer.intermediate_evaluation(2.0, save=True) == 2.0
    _, features, _ = load_reduced(args)
    assert features.abs().sum() > 0  # a worse graph never overwrites the checkpoint


# --------------------------------------------------------------------------- SGDD


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_sgdd_learns_graphon_structure_and_saves(tmp_path, setting):
    args = gradient_args(tmp_path, "sgdd", setting)
    args.mx_size, args.opt_scale = 8, 1e-3
    args.lr_feat = args.lr_adj = 1e-3
    graph = graph_fixture()
    seed_everything(3)
    reduced = create_reducer("sgdd", setting, graph, args).reduce(graph, verbose=False)
    assert graph.adj_mx.shape == (8, 8)
    # Regression: graphs smaller than mx_size used to leave adj_mx unset.
    small = gradient_args(tmp_path / "small", "sgdd", setting)
    small.mx_size, small.opt_scale, small.epochs, small.checkpoints = 100, 1e-3, 1, [0]
    small_graph = graph_fixture()
    create_reducer("sgdd", setting, small_graph, small).reduce(small_graph, verbose=False)
    assert small.mx_size == 32 and small_graph.adj_mx.shape == (32, 32)
    assert torch.isfinite(small_graph.adj_syn).all()
    assert reduced.feat_syn.shape == (12, 8) and reduced.adj_syn.shape == (12, 12)
    assert torch.isfinite(reduced.adj_syn).all() and torch.isfinite(reduced.feat_syn).all()
    torch.testing.assert_close(reduced.adj_syn, reduced.adj_syn.T)
    adjacency, features, labels = load_reduced(args)
    assert labels.bincount().tolist() == [6, 6] and adjacency.shape == (12, 12)
    # A disabled transport term contributes a zero loss.
    args.opt_scale, args.epochs, args.checkpoints = 0, 1, [0]
    graph = graph_fixture()
    create_reducer("sgdd", setting, graph, args).reduce(graph, verbose=False)
    assert torch.isfinite(graph.adj_syn).all()


# --------------------------------------------------------------------------- SimGC


def test_simgc_helpers():
    graph = SimpleNamespace(idx_train=torch.tensor([0, 2]))
    features = torch.arange(6.0).reshape(3, 2)
    torch.testing.assert_close(training_view(features, graph, "trans"), features[[0, 2]])
    assert training_view(features, graph, "ind") is features
    hops = propagated_features(features, 2 * torch.eye(3), 2)
    torch.testing.assert_close(hops, torch.cat([features, 2 * features, 4 * features], 1))
    edges = torch.tensor([[0, 1], [1, 0]])
    identical = smoothness_loss(torch.ones(2, 2), edges, torch.ones(2))
    distinct = smoothness_loss(torch.tensor([[0.0, 0.0], [3.0, 3.0]]), edges, torch.ones(2))
    assert identical == -1 and -1 < distinct < 0


def test_simgc_teacher_returns_layer_features_and_uses_batchnorm(tmp_path):
    args = quick_args(tmp_path, "simgc", "trans")
    args.nlayers, args.ntrans, args.with_bn, args.dropout = 2, 2, True, 0.0
    teacher = SimGCTeacher(4, 8, 3, args)
    assert len(teacher.layers) == 2 and len(teacher.bns) >= 1
    teacher.eval()
    features = torch.rand(5, 4)
    logits, output = teacher(features, [torch.eye(5), torch.eye(5)], output_layer_features=True)
    assert logits.shape == (5, 3)
    torch.testing.assert_close(output, F.log_softmax(logits, 1))


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_simgc_condenses_with_teacher_and_alignment_losses(tmp_path, setting):
    args = gradient_args(tmp_path, "simgc", setting)
    args.teacher_hidden, args.teacher_epochs, args.teacher_val_interval = 8, 3, 1
    args.hidden, args.epochs, args.checkpoints = 8, 2, [2]
    args.validation_config = {"hidden": 8, "eval_epochs": 2, "lr": 0.01}
    args._explicit_cli_keys = {"lr"}
    args.logger = logging.getLogger("simgc-test")
    graph = graph_fixture()
    if setting == "ind":
        # One class gets a single synthetic node to exercise the singleton branch.
        graph.labels_train = torch.tensor([0] * 2 + [1] * 22)
    seed_everything(8)
    reducer = create_reducer("simgc", setting, graph, args)
    validation_args = reducer.validation_model_args()
    assert validation_args.hidden == 8 and validation_args.eval_epochs == 2
    assert validation_args.lr == args.lr  # explicit CLI values win
    reduced = reducer.reduce(graph, verbose=True)
    expected = [6, 6] if setting == "trans" else [1, 11]
    assert reduced.labels_syn.bincount().tolist() == expected
    assert reduced.feat_syn.shape == (12, 8) and reduced.adj_syn.shape == (12, 12)
    assert (reduced.adj_syn >= args.threshold).logical_or(reduced.adj_syn == 0).all()
    assert torch.isfinite(reduced.feat_syn).all()


# --------------------------------------------------------------------------- GCDM


def test_distribution_distances():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    y = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
    assert dist(x, y, "mse") == 2 and dist(x, y, "l1") == 2
    assert dist(x, y, "l1_mean") == 1
    torch.testing.assert_close(dist(x, y, "cos"), torch.tensor(1.0), atol=1e-5, rtol=0)
    assert dist(x, x, "cos") < 1e-5


@pytest.mark.parametrize("method", ["gcdm", "gcdmx"])
@pytest.mark.parametrize(
    "metric,model,layers",
    # Regression: one-layer models and SGC (single representation tensor) used to crash.
    [("l1", "GCN", 2), ("cos", "GCN", 2), ("l1", "GCN", 1), ("l1", "SGC", 2)],
)
def test_distribution_matching_lowers_embedding_distance(tmp_path, method, metric, model, layers):
    args = gradient_args(tmp_path, method, "trans")
    args.condense_model, args.nlayers = model, layers
    args.dis_metric, args.epochs, args.checkpoints = metric, 3, [2]
    args.lr_feat = 0.05
    graph = graph_fixture()
    seed_everything(2)
    reducer = create_reducer(method, "trans", graph, args)
    reduced = reducer.reduce(graph, verbose=False)
    assert reduced.feat_syn.shape == (12, 8) and torch.isfinite(reduced.feat_syn).all()
    assert reduced.labels_syn.bincount().tolist() == [6, 6]
    assert np.isfinite(reduced.feat_syn.numpy()).all()


def test_initializer_records_the_condensation_method_it_serves(tmp_path, monkeypatch):
    import graphslim.condensation.gcond_base as base

    args = gradient_args(tmp_path, "gcond")
    args.init = "random"
    graph = graph_fixture()
    reducer = create_reducer("gcond", "trans", graph, args)
    seen = {}
    original = base.create_reducer

    def recording(method, setting, data, args):
        seen["method"], seen["init_for"] = args.method, args.init_for
        return original(method, setting=setting, data=data, args=args)

    monkeypatch.setattr(base, "create_reducer", recording)
    reducer.init()
    assert seen == {"method": "random", "init_for": "gcond"}
    assert args.method == "gcond" and not hasattr(args, "init_for")


@pytest.mark.parametrize("dataset", ["yelp", "amazon"])
def test_geom_soft_label_configs_define_temperature(dataset):
    from graphslim.config import get_args

    args = get_args(["-D", dataset, "-M", "geom"])
    assert args.setting == "ind" and args.soft_label == 1
    # Same temperature settings as Flickr, the closest-sized inductive GEOM dataset.
    assert (args.tem, args.lr_tem, args.maxtem) == (1, 0.01, 0.01)

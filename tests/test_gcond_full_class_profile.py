"""Actual GCond full-class training, graph selection, and downstream reload."""

import pytest
import torch

from graphslim.dataset import load_reduced
from graphslim.evaluation import Evaluator
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args
from test_legacy_reducer_integration import graph_fixture


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_full_class_gcond_saves_finite_graph_and_trains_gcn(tmp_path, setting):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = quick_args(tmp_path, "gcond", setting)
        args.condense_model = "SGC"
        args.ntrans, args.inner_loop, args.outer_loop = 1, 0, 2
        args.dropout, args.with_bn, args.alpha = 0.0, False, 0.1
        args.dis_metric = "ours"
        args.epochs, args.checkpoints, args.run_inter_eval = 12, [0, 10, 11], 1
        args.validation_only, args.gcond_full_class_gradients = True, True
        args.eval_epochs, args.val_interval = 2, 1
        seed_everything(6)
        graph = graph_fixture()
        reducer = create_reducer("gcond", setting, graph, args)
        reducer.reduce(graph, verbose=False)
        adjacency, features, labels = load_reduced(args)
        assert adjacency.shape == (len(features), len(features))
        assert adjacency.isfinite().all() and features.isfinite().all()
        assert labels.unique().numel() == graph.nclass
        graph.adj_syn, graph.feat_syn, graph.labels_syn = adjacency, features, labels
        validation, test = Evaluator(args).test(graph, "GCN", reduced=True, mode="cross")
        assert 0 <= validation <= 1 and 0 <= test <= 1
    finally:
        torch.set_num_threads(previous)

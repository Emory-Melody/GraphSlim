# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Run small real evaluations and verify selection and diagnostic boundaries."""

import numpy as np
import pytest
import torch

from graphslim.evaluation import Evaluator
from graphslim.models import DOWNSTREAM_MODELS
from graphslim.reduction import create_reducer
from graphslim.sparsification.kcenter_sample import KCenterSample
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("reduced", [False, True])
def test_membership_diagnostic_runs_real_training_on_original_and_reduced_graphs(
    tmp_path, setting, reduced
):
    args = quick_args(tmp_path, "random", setting)
    args.run_eval, args.verbose = 2, True
    graph = tiny_graph()
    seed_everything(11)
    if reduced:
        graph = create_reducer("random", setting, graph, args).reduce(graph)
    evaluator = Evaluator(args)
    result = evaluator.MIA_evaluate(graph, "GCN", verbose=reduced, reduced=reduced)
    assert len(result) == 2 and np.isfinite(result).all()
    assert 0 <= result[0] <= 1 and 0 <= result[1] <= 0.5
    validation, test, membership = evaluator.test(
        graph, "GCN", reduced=reduced, MIA=True, verbose=True
    )
    assert all(0 <= value <= 1 for value in [validation, test, membership])


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_grid_search_fits_real_candidates_and_tests_only_selected_configuration(tmp_path, setting):
    args = quick_args(tmp_path, "random", setting)
    args.run_eval, args.verbose = 2, True
    evaluator = Evaluator(args)
    observed = []
    actual_test = evaluator.test

    def track(*a, **kw):
        result = actual_test(*a, **kw)
        observed.append((args.hidden, kw.get("validation_only", False), result))
        return result

    evaluator.test = track
    score, selected = evaluator.grid_search(tiny_graph(), "GCN", {"hidden": [4, 8]}, reduced=False)
    validation_calls = observed[:4]
    assert all(call[1] for call in validation_calls)
    means = [np.mean([call[2] for call in validation_calls[i : i + 2]]) for i in [0, 2]]
    assert selected == {"hidden": [4, 8][int(np.argmax(means))]}
    assert len(observed) == 6
    assert all(
        hidden == selected["hidden"] and not only_val for hidden, only_val, _ in observed[4:]
    )
    assert args.hidden == selected["hidden"] and np.isfinite(score).all()


def test_cross_model_workflow_runs_all_registered_downstream_models(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    result = Evaluator(args).train_cross(tiny_graph(), grid_search=False, reduced=False)
    assert set(result) == set(DOWNSTREAM_MODELS)
    for row in result.values():
        assert len(row["score"]) == 2 and np.isfinite(row["score"]).all()
        assert 0 <= row["score"][0] <= 1


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_validation_only_nas_evaluation_never_predicts_test_nodes(
    tmp_path, monkeypatch, reduced, verbose
):
    from graphslim.models import GCN

    args = quick_args(tmp_path, "random", "trans")
    graph = tiny_graph()
    if reduced:
        graph = create_reducer("random", "trans", graph, args).reduce(graph)

    def forbidden(*a, **kw):
        pytest.fail("Validation-only NAS must not predict the held-out test graph")

    monkeypatch.setattr(GCN, "predict", forbidden)
    score = Evaluator(args).nas_evaluate(graph, "GCN", reduced=reduced, verbose=verbose)
    assert len(score) == 2 and 0 <= score[0] <= 1


def test_model_based_kcenter_selects_distinct_training_nodes_and_keeps_class_budgets(tmp_path):
    args = quick_args(tmp_path, "kcenter", "trans")
    graph = tiny_graph()
    reducer = KCenterSample("trans", graph, args)
    selected = reducer.select(graph.feat_full)
    assert len(selected) == len(set(selected)) == 8
    assert set(selected).issubset(set(graph.idx_train.tolist()))
    assert graph.labels_full[selected].bincount().tolist() == [4, 4]
    result = reducer.reduce(graph, verbose=True, save=False)
    assert result.feat_syn.shape == (8, 8)
    assert result.labels_syn.bincount().tolist() == [4, 4]

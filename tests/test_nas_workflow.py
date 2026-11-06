"""Actual short architecture searches, persisted choices, and ranking statistics."""

from pathlib import Path

import numpy as np
import pytest
import torch

from graphslim.evaluation.nas_eval import NasEvaluator, load_csv, load_pkl, save_csv
from graphslim.evaluation.utils import calculate_homophily, inference_via_confidence
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


def test_short_architecture_search_saves_and_reloads_actual_validation_winners(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.nas_epochs, args.run_eval = 3, 1
    graph = graph_fixture()
    seed_everything(7)
    graph = create_reducer("random", "trans", graph, args).reduce(graph)
    value = NasEvaluator(args)
    combinations = [(2, 8, 0.1, "relu"), (2, 8, 0.1, "tanh"), (2, 16, 0.2, "relu")]
    value.parameter_combinations = combinations
    value.evaluate_ori(graph)
    value.evaluate_syn(graph)
    assert len(value.results_ori) == len(value.results_syn) == 3
    assert value.best_params_ori == combinations[int(np.argmax(value.results_ori))]
    assert value.best_params_syn == combinations[int(np.argmax(value.results_syn))]
    folder = Path(value.save_path)
    assert load_pkl(folder / "best_params_ori.pkl") == value.best_params_ori
    assert load_pkl(folder / "random_best_params_syn.pkl") == value.best_params_syn
    assert [float(x) for x in load_csv(folder / "results_ori.csv")[0]] == value.results_ori
    restored = NasEvaluator(args)
    restored.test_params_ori(graph)
    result = restored.test_params_syn(graph)
    assert restored.best_params_ori == value.best_params_ori
    assert restored.best_params_syn == value.best_params_syn
    assert 0 <= result <= 1


def test_rank_statistics_handle_ties_and_reload_score_files(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    value = NasEvaluator(args)
    assert value.get_rank([0.9, 0.9, 0.4, 0.8]) == [1, 1, 4, 3]
    original, synthetic = [0.1, 0.3, 0.2], [0.2, 0.6, 0.4]
    save_csv(Path(value.save_path) / "results_ori.csv", original)
    save_csv(Path(value.save_path) / "random_results_syn.csv", synthetic)
    accuracy, ranking = value.cal_pearson()
    assert accuracy == pytest.approx(1) and ranking == pytest.approx(1)
    assert value.results_ori == original and value.results_syn == synthetic


def test_confidence_privacy_diagnostic_distinguishes_separation_and_ties():
    labels = np.array([0, 1, 0, 1])
    train = np.eye(2)[labels] * 0.8 + 0.1
    test = np.eye(2)[labels] * -0.8 + 0.9
    assert inference_via_confidence(train, test, labels, labels) == 1
    tied = np.full((4, 2), 0.5)
    assert inference_via_confidence(tied, tied, labels, labels) == 0.5
    adjacency = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert calculate_homophily(labels, adjacency) == 0.5

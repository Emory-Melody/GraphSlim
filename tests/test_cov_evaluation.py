"""Evaluator workflows, diagnostic utilities, and graph-property statistics."""

import logging
from types import SimpleNamespace

import matplotlib
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

from graphslim.evaluation import Evaluator
from graphslim.evaluation import eval_agent
from graphslim.evaluation.graph_property import PropertyEvaluator, graph_statistics
from graphslim.evaluation.utils import (
    calc_f1,
    calculate_homophily,
    evaluate,
    getsize_mb,
    verbose_time_memory,
)
from graphslim.models import DOWNSTREAM_MODELS
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args, tiny_graph

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


# ------------------------------------------------------------------ Evaluator


def test_evaluate_reports_mean_and_std_of_recorded_runs(tmp_path, capsys):
    args = quick_args(tmp_path, "random", "trans")
    args.run_eval = 2
    graph = tiny_graph()
    seed_everything(0)
    graph = create_reducer("random", "trans", graph, args).reduce(graph)
    evaluator = Evaluator(args)
    mean, std = evaluator.evaluate(graph, "GCN", verbose=True, reduced=True)
    tests = [run["test"] for run in evaluator.last_results]
    assert [run["seed"] for run in evaluator.last_results] == [args.seed, args.seed + 1]
    assert mean == pytest.approx(np.mean(tests)) and std == pytest.approx(np.std(tests))
    assert all(0 <= run["validation"] <= 1 for run in evaluator.last_results)
    assert "Test Mean Accuracy" in capsys.readouterr().out
    del args.logger
    assert Evaluator(args).evaluate(graph, "SGC", verbose=False, reduced=False)[0] >= 0


def test_training_noise_is_applied_and_cached_before_fitting(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.attack, args.ptb_r = "edge_drop", 0.5
    graph = tiny_graph()
    edges = graph.adj_full.nnz
    validation, test = Evaluator(args).test(graph, "GCN", verbose=True, reduced=False)
    assert 0 <= validation <= 1 and 0 <= test <= 1
    cached = list((tmp_path / "corrupt_graph" / "edge_drop").glob("*.json"))
    assert len(cached) == 1
    assert graph.adj_full.nnz == edges


def test_cross_model_grid_search_covers_every_architecture(tmp_path, monkeypatch, caplog):
    args = quick_args(tmp_path, "random", "trans")
    args.dataset, args.verbose = "reddit", False
    args.logger = logging.getLogger("cov-evaluation")
    searched = {}

    def first_only(grid):
        searched[len(searched)] = grid
        return [{key: values[0] for key, values in grid.items()}]

    monkeypatch.setattr(eval_agent, "ParameterGrid", first_only)
    graph = tiny_graph()
    seed_everything(0)
    graph = create_reducer("random", "trans", graph, args).reduce(graph)
    with caplog.at_level(logging.INFO, logger="cov-evaluation"):
        results = Evaluator(args).train_cross(graph, grid_search=True, reduced=True)
    assert set(results) == set(DOWNSTREAM_MODELS)
    assert searched[6]["hidden"] == [8, 16]  # GAT grid narrowed for Reddit
    for row in results.values():
        assert 0 <= row["score"][0] <= 1 and row["score"][1] >= 0
        assert set(row["params"]) <= {
            "hidden",
            "lr",
            "weight_decay",
            "dropout",
            "ntrans",
            "alpha",
            "trans_num_layers",
            "trans_weight_decay",
            "trans_dropout",
        }
    assert caplog.text.count("Best ") == len(DOWNSTREAM_MODELS)


def test_tsne_visualization_saves_a_pdf(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = quick_args(tmp_path, "random", "trans")
    args.dataset = "toy"
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(12, 3, generator=generator)
    labels = torch.arange(12) % 2
    Evaluator(args).tsne_vis(features, labels, features[:4], labels[:4])
    name = f"tsne_visualization_random_toy_{args.reduction_rate}.pdf"
    assert (tmp_path / name).stat().st_size > 0
    assert name in capsys.readouterr().out


# ------------------------------------------------------------------ utilities


def test_homophily_accepts_dense_csr_and_coo_inputs():
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    labels = np.array([[0], [0], [1]])
    for value in (adjacency, sp.csr_matrix(adjacency), sp.coo_matrix(adjacency)):
        assert calculate_homophily(labels, value) == pytest.approx(0.5)


def test_size_in_megabytes_counts_every_storage_type():
    dense = torch.zeros(4, 4)
    csr = sp.csr_matrix(np.eye(4))
    sparse = SparseTensor.from_dense(torch.eye(4))
    array = np.zeros(8, dtype=np.float64)
    megabyte = 1024 * 1024
    assert getsize_mb([dense]) == pytest.approx(64 / megabyte)
    assert getsize_mb([array]) == pytest.approx(64 / megabyte)
    assert getsize_mb([csr]) == pytest.approx(2 * 4 * 8 / megabyte)
    assert getsize_mb([sparse]) == pytest.approx((8 + 8 + 4) * 4 / megabyte)


class Reducer:
    @verbose_time_memory
    def run(self, data, verbose=False, setting=None):
        return "done"


def test_timing_decorator_reports_storage_and_fills_missing_synthetic_fields(capsys):
    full = tiny_graph()
    assert Reducer().run(full, verbose=True, setting="trans") == "done"
    assert full.feat_syn is full.feat_full
    assert full.labels_syn is full.labels_train
    assert torch.equal(full.adj_syn, torch.eye(16))
    train = SimpleNamespace(
        feat_train=torch.zeros(2, 2), adj_train=torch.eye(2), labels_train=torch.zeros(2)
    )
    Reducer().run(data=train, verbose=True)
    assert train.feat_syn is train.feat_train
    out = capsys.readouterr().out
    assert out.count("Function Time") == 4 and "Condensed graph" in out
    with pytest.raises(ValueError, match="'data' as an argument"):
        Reducer().run(SimpleNamespace(), verbose=True)


def test_f1_scores_for_class_scores_and_sigmoid_probabilities():
    micro, macro = calc_f1(np.array([0, 1, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]))
    assert micro == pytest.approx(2 / 3) and macro == pytest.approx(2 / 3)
    probabilities = np.array([[0.7, 0.2], [0.4, 0.9]])
    micro, macro = calc_f1(np.array([[1, 0], [0, 1]]), probabilities, is_sigmoid=True)
    assert micro == macro == 1.0
    assert probabilities.tolist() == [[1, 0], [0, 1]]


def test_graphsaint_evaluation_prints_f1_scores(capsys):
    args = SimpleNamespace(dataset="flickr")
    evaluate(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), torch.tensor([0, 1]), args)
    assert "F1-micro= 1.0000 F1-macro= 1.0000" in capsys.readouterr().out
    evaluate(torch.tensor([[0.9, 0.1], [0.3, 0.6]]), torch.tensor([[1, 0], [0, 1]]), args)
    assert "F1-micro= 1.0000" in capsys.readouterr().out


# ------------------------------------------------------------ graph property


def test_graph_statistics_validate_dimensions_and_skip_structure():
    with pytest.raises(ValueError, match="dimensions"):
        graph_statistics(torch.eye(3), torch.zeros(2, 2), torch.zeros(3))
    result = graph_statistics(torch.eye(4), torch.eye(4), torch.tensor([0, 0, 1, 1]), False)
    assert set(result) == {"nodes", "davies_bouldin"} and result["nodes"] == 4


def test_large_graph_uses_sparse_eigensolver_for_spectral_radius():
    graph = nx.cycle_graph(80)
    adjacency = nx.to_scipy_sparse_array(graph, format="csr").astype(np.float32)
    labels = np.arange(80) % 2
    result = graph_statistics(adjacency, np.random.default_rng(0).random((80, 3)), labels)
    assert result["spectral_radius"] == pytest.approx(4.0, abs=1e-6)
    assert result["edges"] == 80 and result["homophily"] == 0.0


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_property_evaluator_reports_original_and_batched_graphs(tmp_path, setting):
    args = quick_args(tmp_path, "random", setting)
    args.with_structure = 1
    graph = tiny_graph()
    report = PropertyEvaluator(args).evaluate(graph, reduced=False)
    assert report["nodes"] == (16 if setting == "ind" else 32)
    evaluator = PropertyEvaluator(args)
    evaluator.feat = torch.rand(4, 3)
    evaluator.label = torch.tensor([0, 1, 0, 1])
    evaluator.adj = torch.stack([torch.eye(4), torch.ones(4, 4)])
    reports = evaluator.graph_property_no_structure()
    assert isinstance(reports, list) and len(reports) == 2
    assert all(set(r) == {"nodes", "davies_bouldin"} for r in reports)


def test_graph_without_edges_has_undefined_structural_statistics():
    one_hot = torch.eye(2)[torch.tensor([0, 1, 0, 1])]
    result = graph_statistics(sp.csr_matrix((4, 4)), torch.rand(4, 3), one_hot)
    assert result["edges"] == 0 and result["spectral_radius"] == 0.0
    assert result["laplacian_trace"] is None and result["homophily"] is None
    assert result["davies_bouldin"] is not None


def test_property_evaluator_reads_saved_reduced_graph(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.with_structure = 1
    graph = tiny_graph()
    seed_everything(0)
    graph = create_reducer("random", "trans", graph, args).reduce(graph)
    report = PropertyEvaluator(args).evaluate(graph, reduced=True)
    assert report["nodes"] == len(graph.feat_syn)
    assert {"density", "clustering", "homophily"} <= set(report)


def test_cross_grid_search_on_original_graph(tmp_path, monkeypatch):
    args = quick_args(tmp_path, "random", "trans")
    args.verbose = False
    monkeypatch.setattr(
        eval_agent, "ParameterGrid", lambda grid: [{k: v[-1] for k, v in grid.items()}]
    )
    evaluator = Evaluator(args)
    graph = tiny_graph()
    results = evaluator.train_cross(graph, grid_search=True, reduced=False)
    assert results["GAT"]["params"]["hidden"] == 64
    assert all(0 <= row["score"][0] <= 1 for row in results.values())


def test_nas_evaluator_uses_in_memory_choices_and_scores(tmp_path):
    from graphslim.evaluation.nas_eval import NasEvaluator

    args = quick_args(tmp_path, "random", "trans")
    args.nas_epochs = 2
    graph = tiny_graph()
    seed_everything(0)
    graph = create_reducer("random", "trans", graph, args).reduce(graph)
    value = NasEvaluator(args)
    value.best_params_ori = value.best_params_syn = (2, 8, 0.1, "relu")
    value.test_params_ori(graph)
    assert 0 <= value.test_params_syn(graph) <= 1
    assert (args.nlayers, args.hidden, args.alpha, args.activation) == (2, 8, 0.1, "relu")
    value.results_ori, value.results_syn = [0.1, 0.2, 0.3], [0.3, 0.2, 0.1]
    accuracy, rank = value.cal_pearson()
    assert accuracy == pytest.approx(-1) and rank == pytest.approx(-1)

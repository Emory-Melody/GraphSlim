import json
import statistics
from types import SimpleNamespace

import torch
import pytest

from benchmark.rerun_nr import SEEDS, aggregate_cell, job_path, plan_jobs, write_json
from graphslim.condensation.simgc import training_view
from graphslim.config import get_args
from graphslim.evaluation import Evaluator
from graphslim.models import GCN


def test_nr_roster_is_complete_and_missing_configuration_is_explicit():
    jobs = plan_jobs()
    assert len(jobs) == 315
    assert len({job["id"] for job in jobs}) == 315
    blocked = [job for job in jobs if job["blocked_reason"]]
    assert len(blocked) == 9
    assert {(job["method"], job["dataset"]) for job in blocked} == {("gecc", "yelp")}


def test_inductive_simgc_aggregation_does_not_use_global_training_indices():
    features = torch.tensor([[1.0], [2.0]])
    graph = SimpleNamespace(idx_train=torch.tensor([7, 10]))
    assert torch.equal(training_view(features, graph, "ind"), features)
    full = torch.arange(12).view(-1, 1)
    assert torch.equal(training_view(full, graph, "trans"), torch.tensor([[7], [10]]))


@pytest.mark.parametrize("dataset", ["cora", "yelp"])
def test_gcn_and_evaluator_accept_tensor_accuracy_and_float_f1(tmp_path, dataset):
    args = get_args(
        [
            "-D",
            dataset,
            "-M",
            "clustering",
            "-G",
            "-1",
            "--save_path",
            str(tmp_path),
            "--eval_epochs",
            "2",
        ]
    )
    args.setting = "ind"
    graph = SimpleNamespace(
        nclass=2,
        feat_full=torch.tensor([[1.0], [2.0]]),
        feat_syn=torch.tensor([[1.0], [2.0]]),
        feat_train=torch.tensor([[1.0], [2.0]]),
        feat_val=torch.tensor([[1.0], [2.0]]),
        feat_test=torch.tensor([[1.0], [2.0]]),
        labels_train=torch.tensor([0, 1]),
        labels_val=torch.tensor([0, 1]),
        labels_test=torch.tensor([0, 1]),
        labels_syn=torch.tensor([0, 1]),
        adj_syn=torch.eye(2),
        adj_train=torch.eye(2),
        adj_val=torch.eye(2),
        adj_test=torch.eye(2),
        idx_val=torch.arange(2),
        idx_test=torch.arange(2),
    )
    model = GCN(1, 8, 2, args, mode="eval")
    assert isinstance(model.fit_with_val(graph, train_iters=2, setting="ind", reduced=True), float)
    assert isinstance(model.test(graph, setting="ind"), float)
    val, test = Evaluator(args).test(graph, "GCN", verbose=False, reduced=True)
    assert 0 <= val <= 1 and 0 <= test <= 1


def record_results(tmp_path, jobs, source="frozen"):
    scores = []
    for job in jobs:
        values = [0.5 + job["seed"] * 0.01 + i * 0.001 for i in range(10)]
        scores.extend(values)
        write_json(
            job_path(tmp_path, job) / "result.json",
            {
                "source_sha256": source,
                "data": {"hashes": {"train": "fixed"}},
                "evaluations": [{"test": value} for value in values],
            },
        )
    return scores


def test_partial_reductions_are_never_published_as_scores(tmp_path):
    jobs = plan_jobs(["cora"], ["clustering"])[:3]
    # Sort by seed explicitly; the planner interleaves methods across seeds.
    jobs = [job for job in jobs if job["rate"] == "0.1"]
    assert [job["seed"] for job in jobs] == list(SEEDS)
    record_results(tmp_path, jobs[:1])
    result = aggregate_cell(tmp_path, jobs, "frozen")
    assert result == {"status": "pending", "value": "PENDING (1/3)"}
    assert "mean" not in result


def test_local_uncertainty_is_recomputed_from_all_raw_scores(tmp_path):
    jobs = plan_jobs(["cora"], ["clustering"])[:3]
    scores = record_results(tmp_path, jobs)
    result = aggregate_cell(tmp_path, jobs, "frozen")
    assert result["n"] == 30
    assert result["mean"] == statistics.mean(scores)
    assert result["std"] == statistics.pstdev(scores)


def test_mixed_sources_or_splits_cannot_be_aggregated(tmp_path):
    jobs = plan_jobs(["cora"], ["clustering"])[:3]
    record_results(tmp_path, jobs)
    assert aggregate_cell(tmp_path, jobs, "different")["value"] == "INVALID"
    path = job_path(tmp_path, jobs[-1]) / "result.json"
    record = json.loads(path.read_text())
    record["data"]["hashes"]["train"] = "different"
    write_json(path, record)
    assert aggregate_cell(tmp_path, jobs, "frozen")["value"] == "INVALID"

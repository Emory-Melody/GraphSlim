"""Exercise the artifact writer and separate reduction/evaluation seeds."""

from types import SimpleNamespace
import json

import torch

from benchmark import lightweight_downstream as runner
from test_lightweight_methods import tiny_graph


def test_runner_writes_serializable_complete_roster_with_distinct_eval_seeds(tmp_path, monkeypatch):
    import graphslim.dataset
    import benchmark.verify_lightweight

    monkeypatch.setattr(graphslim.dataset, "get_dataset", lambda *args: tiny_graph())
    monkeypatch.setattr(
        benchmark.verify_lightweight,
        "verify",
        lambda *args: {"status": "stubbed reference lookup; training remains real"},
    )
    options = SimpleNamespace(
        run_dir=tmp_path,
        datasets=["cora"],
        methods=["cgc"],
        conditions=["clean"],
        seeds=[1],
        eval_seeds=[0, 2],
        rate=0.5,
        noise_rate=0.1,
        profile="smoke",
        threads=1,
        references=tmp_path,
    )
    manifest = runner.initialize(options)
    summary = runner.execute(options, manifest)
    assert summary["complete"] == 1 and summary["failed"] == 0
    record = json.loads((tmp_path / "jobs/cora__cgc__clean__s1/result.json").read_text())
    assert {(value["model"], value["seed"]) for value in record["evaluations"]} == {
        (model, seed) for model in runner.MODELS for seed in (0, 2)
    }
    assert len(record["inference_noise"]) == 70
    assert len(record["nas"]["synthetic_validation"]) == 2
    assert len(record["checkpoint_sha256"]) == 3


def test_property_statistics_handle_larger_empty_adjacency():
    from graphslim.evaluation.graph_property import graph_statistics

    statistics = graph_statistics(torch.eye(70), torch.rand(70, 4), torch.arange(70) % 2)
    assert statistics["spectral_radius"] == 0
    assert statistics["laplacian_trace"] is None
    assert statistics["homophily"] is None

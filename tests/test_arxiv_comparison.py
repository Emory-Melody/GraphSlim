"""Large-graph comparisons require paired inputs, full seeds, and measured quality."""

import json
from types import SimpleNamespace

import pytest

from benchmark.compare_arxiv import freeze, jobs, summarize, write_json


def trial(tmp_path):
    roster = jobs(["clustgdd"], ["0.005"], [1, 2, 3])
    write_json(
        tmp_path / "manifest.json",
        dict(
            methods=["clustgdd"],
            rates=["0.005"],
            seeds=[1, 2, 3],
            jobs=roster,
            input_sha256={"x": "common"},
            gpu={"uuid": "GPU-test"},
            data={"nodes": 169343},
            protocol={},
            references={},
        ),
    )
    for job in roster:
        record = dict(
            job,
            status="complete",
            input_sha256={"x": "common"},
            synthetic_nodes=454,
            environment={"gpu_uuid": "GPU-test"},
            reduction_seconds=job["seed"] * (2 if job["backend"] == "author" else 1),
            memory={"peak_cuda_allocated_bytes": 2**30, "peak_sampled_rss_bytes": 2**29},
            evaluations=[{"test_accuracy": 0.7 if job["backend"] == "author" else 0.72}],
        )
        write_json(tmp_path / "jobs" / job["id"] / "result.json", record)
    return tmp_path / "jobs" / "clustgdd__r0.005__s3__graphslim" / "result.json"


def test_roster_covers_full_arxiv_rates_and_alternates_backend_order():
    roster = jobs()
    assert len(roster) == len({j["id"] for j in roster}) == 36
    assert {j["dataset"] for j in roster} == {"ogbn-arxiv"}
    assert {j["rate"] for j in roster} == {"0.001", "0.005", "0.01"}
    for seed in [1, 2, 3]:
        selected = [
            j for j in roster if (j["method"], j["rate"], j["seed"]) == ("cgc", "0.001", seed)
        ]
        assert [j["backend"] for j in selected] == (
            ["author", "graphslim"] if seed % 2 else ["graphslim", "author"]
        )


def test_all_reductions_and_quality_are_needed_before_numeric_cell(tmp_path):
    path = trial(tmp_path)
    record = json.loads(path.read_text())
    record["status"] = "reduced"
    write_json(path, record)
    summary = summarize(tmp_path)
    assert summary["cells"][1]["status"] == "unavailable"
    assert "reduction_seconds" not in summary["cells"][1]
    assert all("author_over_graphslim_time" not in c for c in summary["cells"])


def test_ratios_and_one_sided_quality_are_computed_from_complete_pairs(tmp_path):
    trial(tmp_path)
    summary = summarize(tmp_path)
    assert [c["reduction_seconds"]["median"] for c in summary["cells"]] == [4, 2]
    assert all(c["author_over_graphslim_time"] == 2 for c in summary["cells"])
    assert all(c["relative_accuracy_shortfall_percent"] == 0 for c in summary["cells"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", {"x": "different"}),
        ("synthetic_nodes", 453),
        ("environment", {"gpu_uuid": "GPU-other"}),
    ],
)
def test_mismatched_inputs_budgets_or_devices_never_get_ratios(tmp_path, field, value):
    path = trial(tmp_path)
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(AssertionError):
        summarize(tmp_path)


def test_partial_evidence_cannot_be_frozen_for_publication(tmp_path):
    path = trial(tmp_path)
    record = json.loads(path.read_text())
    record["status"] = "running"
    write_json(path, record)
    snapshot = tmp_path / "public.json"
    with pytest.raises(AssertionError, match="Incomplete cells"):
        freeze(SimpleNamespace(run_dir=tmp_path, snapshot=snapshot))
    assert not snapshot.exists()

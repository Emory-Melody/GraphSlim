# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import json

import pytest

from benchmark.compare_implementations import summarize, write_json


def trial(tmp_path, *, evaluations=2):
    manifest = dict(
        methods=["gcgp"],
        datasets=["cora"],
        seeds=[1, 2, 3],
        eval_seeds=list(range(evaluations)),
        protocol={"scope": "controlled"},
    )
    write_json(tmp_path / "manifest.json", manifest)
    for seed in manifest["seeds"]:
        for backend in ("author", "graphslim"):
            record = dict(
                status="complete",
                input_sha256={"x": "same", "y": "same"},
                synthetic_nodes=70,
                device="cpu",
                reduction_seconds=seed * (2 if backend == "author" else 1),
                memory={"peak_sampled_rss_bytes": 2**20, "additional_sampled_rss_bytes": 2**19},
                evaluations=[dict(test_accuracy=0.7 + 0.01 * i) for i in manifest["eval_seeds"]],
            )
            write_json(
                tmp_path / "jobs" / f"gcgp__cora__s{seed}__{backend}" / "result.json", record
            )


def test_aggregates_all_fixed_seeds_and_fits(tmp_path):
    trial(tmp_path)
    rows = summarize(tmp_path)
    assert len(rows) == 2
    assert all(r["evaluations"] == 6 for r in rows)
    assert all(r["accuracy_mean"] == pytest.approx(70.5) for r in rows)
    assert all(r["author_over_graphslim_time"] == 2 for r in rows)
    assert [r["reduction_seconds_median"] for r in rows] == [4, 2]


def test_incomplete_quality_never_yields_a_paired_speedup(tmp_path):
    trial(tmp_path)
    path = tmp_path / "jobs/gcgp__cora__s3__graphslim/result.json"
    record = json.loads(path.read_text())
    record["evaluations"].pop()
    write_json(path, record)
    rows = summarize(tmp_path)
    assert rows[1]["status"] == "incomplete"
    assert all("author_over_graphslim_time" not in r for r in rows)
    assert "accuracy_mean" not in rows[1]


@pytest.mark.parametrize(
    "field,value", [("input_sha256", {"x": "different"}), ("synthetic_nodes", 69)]
)
def test_rejects_input_or_budget_mismatch(tmp_path, field, value):
    trial(tmp_path)
    path = tmp_path / "jobs/gcgp__cora__s2__graphslim/result.json"
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError, match="Mismatched"):
        summarize(tmp_path)

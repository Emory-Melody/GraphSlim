# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Guard numerical-reference-only reporting and its missing-reference boundary."""

import json

from benchmark.plot_reproduction_coverage import SOURCE
from benchmark.plot_reproduction_coverage import reviewed_data


def test_shortfall_thresholds_preserve_the_frozen_roster():
    data = reviewed_data()
    assert data["metric"] == "mean_shortfall_percent"
    assert data["formula"] == "mean_r(100 * max(0, reference_r - reproduced_r) / reference_r)"
    assert data["total_pairs"] == 110
    assert data["method_count"] == 17
    assert data["numeric_reference_pairs"] == 110
    assert [c["threshold_percent"] for c in data["columns"]] == [1, 2, 5, 10]
    assert [c["count"] for c in data["columns"]] == [65, 85, 107, 110]
    assert [c["measured_count"] for c in data["columns"]] == [65, 85, 107, 110]


def test_missing_references_never_enter_reproduction_counts():
    data = reviewed_data()
    source = json.loads(SOURCE.read_text())
    assert all(p["reference_status"] == "numeric_reference" for p in source["pairs"])
    assert not any(p["credited_by_policy"] or p["method"] == "clustering" for p in source["pairs"])
    assert not any(p["method"] in ("cent_p", "gdem") for p in source["pairs"])
    assert all(s["kind"] != "policy_credit" for s in data["segments"])
    keys = [key for s in data["segments"] for key in s["pairs"]]
    assert len(keys) == len(set(keys)) == 110


def test_paper_bilateral_errors_use_the_same_numerical_reference_roster():
    data = reviewed_data("mean_relative_gap_percent")
    assert data["total_pairs"] == 110
    assert [c["count"] for c in data["columns"]] == [39, 64, 96, 103]
    assert len(data["outside_largest_threshold"]) == 7
    assert all(c["count"] == c["measured_count"] for c in data["columns"])


def test_roster_drops_only_centp_and_gdem_from_the_124_pair_parent():
    source = json.loads(SOURCE.read_text())
    revision = source["roster_revision_20261106"]
    parent = json.loads((SOURCE.parents[2] / revision["parent_snapshot"]).read_text())
    excluded = {"cent_p", "gdem"}
    assert len(parent["pairs"]) == 124 and set(revision["reasons"]) == excluded
    rerun = set(revision["reruns_after_fix"]["pairs"])
    kept = [p for p in parent["pairs"] if p["method"] not in excluded]
    assert [(p["method"], p["dataset"]) for p in source["pairs"]] == [
        (p["method"], p["dataset"]) for p in kept
    ]
    assert [p for p in source["pairs"] if f"{p['method']}/{p['dataset']}" not in rerun] == [
        p for p in kept if f"{p['method']}/{p['dataset']}" not in rerun
    ]
    assert rerun == {
        f"{method}/{dataset}"
        for method, dataset in [
            ("herding", "flickr"),
            ("herding", "reddit"),
            ("herding", "yelp"),
            ("kcenter", "flickr"),
            ("kcenter", "yelp"),
        ]
    }
    removed = {(p["method"], p["dataset"]) for p in parent["pairs"] if p["method"] in excluded}
    assert {(p["method"], p["dataset"]) for p in revision["removed_pairs"]} == removed
    assert len(removed) == 14


def test_new_gc4nc_pairs_preserve_the_legacy_roster_and_shared_source_boundary():
    source = json.loads(SOURCE.read_text())
    revision = source["roster_revision_20261106"]
    parent = json.loads((SOURCE.parents[2] / revision["parent_snapshot"]).read_text())
    legacy = json.loads((SOURCE.parents[2] / parent["cohort_expansion"]["parent"]).read_text())
    kept = [p for p in legacy["pairs"] if p["method"] not in ("cent_p", "gdem")]
    keys = [(p["method"], p["dataset"]) for p in source["pairs"]]
    assert keys[: len(kept)] == [(p["method"], p["dataset"]) for p in kept]
    added = source["pairs"][len(kept) :]
    assert {(p["method"], p["dataset"]) for p in added} == {
        ("t_spanner", dataset) for dataset in ["ogbn-arxiv", "flickr", "reddit", "yelp"]
    }
    assert all(not p["independent_replication"] and not p["credited_by_policy"] for p in added)
    assert all(p["mean_shortfall_percent"] < 1 for p in added)
    assert all(p["score_metric"] == "AP" for p in added if p["dataset"] == "yelp")


def frozen_verifier():
    import importlib.util

    path = SOURCE.parent / "reproducibility_20261103" / "verify.py"
    spec = importlib.util.spec_from_file_location("frozen_reproducibility_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shortfall_does_not_imply_two_sided_agreement_or_cancel_other_settings():
    import statistics

    verify = frozen_verifier()
    assert verify.errors(80, 88) == (0, 10, 8)
    settings = [verify.errors(80, 72), verify.errors(80, 88)]
    mean_shortfall = statistics.mean(e[0] for e in settings)
    assert mean_shortfall == 5
    assert not mean_shortfall < 5  # The boundary is strict, not inclusive.
    assert statistics.mean(e[1] for e in settings) == 10


def test_fixed_artifact_recomputes_historical_policy_and_both_current_metrics():
    verify = frozen_verifier()
    summary = verify.verify(SOURCE.parent / "reproducibility_20261103")
    views = summary["cohort_results"]
    assert views["historical_with_policy"]["mean_relative_gap_percent"]["5"]["count"] == 111
    assert views["numerical_only_116"]["mean_relative_gap_percent"]["5"]["count"] == 104
    assert views["paper_124"]["mean_relative_gap_percent"]["5"]["count"] == 112
    assert views["paper_124"]["mean_shortfall_percent"]["5"]["count"] == 122
    assert (
        summary["uncorrected_table9_sensitivity"]["mean_relative_gap_percent"]["5"]["count"] == 110
    )


def test_current_artifact_recomputes_the_110_pair_roster_from_the_unchanged_base():
    import importlib.util

    path = SOURCE.parent / "reproducibility_20261106" / "verify.py"
    spec = importlib.util.spec_from_file_location("current_reproducibility_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.verify(path.parent)
    assert (summary["pair_count"], summary["method_count"], summary["setting_count"]) == (
        110,
        17,
        313,
    )
    views = summary["cohort_results"]
    assert views["paper_110"]["mean_shortfall_percent"]["5"]["count"] == 107
    assert views["paper_110"]["mean_relative_gap_percent"]["5"]["count"] == 96
    assert views["without_shared_observations_106"]["mean_shortfall_percent"]["5"]["count"] == 103
    assert summary["replaced_settings"] == 15
    snapshot = json.loads(SOURCE.read_text())
    for metric in ("mean_shortfall_percent", "mean_relative_gap_percent"):
        for threshold, expected in snapshot["thresholds"][metric].items():
            assert views["paper_110"][metric][threshold]["count"] == expected["count"]

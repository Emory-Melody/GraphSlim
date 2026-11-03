"""Guard numerical-reference-only reporting and its missing-reference boundary."""

import json

from benchmark.plot_reproduction_coverage import SOURCE
from benchmark.plot_reproduction_coverage import reviewed_data


def test_shortfall_thresholds_preserve_the_frozen_roster():
    data = reviewed_data()
    assert data["metric"] == "mean_shortfall_percent"
    assert data["formula"] == "mean_r(100 * max(0, reference_r - reproduced_r) / reference_r)"
    assert data["total_pairs"] == 124
    assert data["method_count"] == 19
    assert data["numeric_reference_pairs"] == 124
    assert [c["threshold_percent"] for c in data["columns"]] == [1, 2, 5, 10]
    assert [c["count"] for c in data["columns"]] == [72, 94, 122, 124]
    assert [c["measured_count"] for c in data["columns"]] == [72, 94, 122, 124]


def test_missing_references_never_enter_reproduction_counts():
    data = reviewed_data()
    source = json.loads(SOURCE.read_text())
    assert all(p["reference_status"] == "numeric_reference" for p in source["pairs"])
    assert not any(p["credited_by_policy"] or p["method"] == "clustering" for p in source["pairs"])
    assert all(s["kind"] != "policy_credit" for s in data["segments"])
    keys = [key for s in data["segments"] for key in s["pairs"]]
    assert len(keys) == len(set(keys)) == 124


def test_paper_bilateral_errors_use_the_same_numerical_reference_roster():
    data = reviewed_data("mean_relative_gap_percent")
    assert data["total_pairs"] == 124
    assert [c["count"] for c in data["columns"]] == [43, 70, 112, 117]
    assert len(data["outside_largest_threshold"]) == 7
    assert all(c["count"] == c["measured_count"] for c in data["columns"])


def test_new_gc4nc_pairs_preserve_the_legacy_roster_and_shared_source_boundary():
    source = json.loads(SOURCE.read_text())
    parent = json.loads((SOURCE.parents[2] / source["cohort_expansion"]["parent"]).read_text())
    assert source["pairs"][:116] == parent["pairs"]
    added = source["pairs"][116:]
    assert {(p["method"], p["dataset"]) for p in added} == {
        (method, dataset) for method in ["t_spanner", "gdem"]
        for dataset in ["ogbn-arxiv", "flickr", "reddit", "yelp"]
    }
    assert all(not p["independent_replication"] and not p["credited_by_policy"] for p in added)
    assert all(p["mean_shortfall_percent"] < 1 for p in added)
    assert all(p["score_metric"] == "AP" for p in added if p["dataset"] == "yelp")


def frozen_verifier():
    import importlib.util
    path = SOURCE.parent / 'reproducibility_20261103' / 'verify.py'
    spec = importlib.util.spec_from_file_location('frozen_reproducibility_verifier', path)
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
    summary = verify.verify(SOURCE.parent / 'reproducibility_20261103')
    views = summary['cohort_results']
    assert views['historical_with_policy']['mean_relative_gap_percent']['5']['count'] == 111
    assert views['numerical_only_116']['mean_relative_gap_percent']['5']['count'] == 104
    assert views['paper_124']['mean_relative_gap_percent']['5']['count'] == 112
    assert views['paper_124']['mean_shortfall_percent']['5']['count'] == 122
    assert summary['uncorrected_table9_sensitivity']['mean_relative_gap_percent']['5']['count'] == 110

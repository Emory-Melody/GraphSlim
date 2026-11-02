"""Guard the reviewed bilateral contract and its missing-reference boundary."""

from benchmark.plot_reproduction_coverage import reviewed_data


def test_bilateral_thresholds_preserve_the_frozen_roster():
    data = reviewed_data()
    assert data["metric"] == "mean_relative_gap_percent"
    assert data["total_pairs"] == 123
    assert data["numeric_reference_pairs"] == 116
    assert [c["threshold_percent"] for c in data["columns"]] == [1, 2, 5, 10]
    assert [c["count"] for c in data["columns"]] == [42, 69, 111, 116]
    assert [c["measured_count"] for c in data["columns"]] == [35, 62, 104, 109]


def test_completion_credit_is_separate_from_a_measured_error():
    data = reviewed_data()
    credit = next(s for s in data["segments"] if s["kind"] == "policy_credit")
    assert credit["count"] == 7
    assert credit["measured_relative_error"] is None
    assert all(c["policy_credit_count"] == 7 for c in data["columns"])
    measured = [s for s in data["segments"] if s["kind"] != "policy_credit"]
    keys = [key for s in measured for key in s["pairs"]]
    assert len(keys) == len(set(keys)) == 109
    assert not set(keys) & set(credit["pairs"])

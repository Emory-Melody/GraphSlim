"""Render README comparison tables from frozen evidence, without running benchmarks.

Run ``python -m benchmark.render_readme_highlights`` to update both languages,
or append ``--check`` to verify them. The coverage snapshot includes its fixed
roster, formulas, policy credits, reference corrections, and source hashes.
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "benchmark/results/reproduction_coverage_20261101.json"
COMPARISON = ROOT / "benchmark/results/implementation_comparison_20261031.json"


def load_coverage():
    data = json.loads(COVERAGE.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    assert len({(p["method"], p["dataset"]) for p in pairs}) == data["total_pairs"]
    assert sum(p["complete_result"] for p in pairs) == data["complete_pairs"]
    assert sum(p["credited_by_policy"] for p in pairs) == data["policy_credits"]["pairs"]
    for metric, thresholds in data["thresholds"].items():
        for threshold, expected in thresholds.items():
            count = 0
            for pair in pairs:
                value = pair[metric]
                if pair["credited_by_policy"]:
                    assert value is None
                    count += 1
                else:
                    assert math.isfinite(value) and value >= 0
                    count += pair["complete_result"] and value < float(threshold)
            assert count == expected["count"]
            assert expected["denominator"] == data["total_pairs"]
            assert math.isclose(expected["percent"], 100 * count / data["total_pairs"])
    return data


def coverage_table(data, language):
    if language == "en":
        header = "| Criterion | <1% | <2% | <5% |"
        labels = ("Relative shortfall (primary)", "Absolute relative gap")
    else:
        header = "| 判定口径 | <1% | <2% | <5% |"
        labels = ("单边相对短缺（主口径）", "双边绝对相对差距")
    lines = [header, "| --- | ---: | ---: | ---: |"]
    for metric, label in zip(("mean_shortfall_percent", "mean_relative_gap_percent"), labels):
        cells = []
        for threshold in ("1", "2", "5"):
            row = data["thresholds"][metric][threshold]
            cells.append(f"{row['count']}/{row['denominator']} ({row['percent']:.2f}%)")
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def speed_table(language):
    data = json.loads(COMPARISON.read_text(encoding="utf-8"))
    rows = data["rows"]
    expected = {
        (m, d, b)
        for m in ("cgc", "clustgdd", "gcgp")
        for d in ("cora", "citeseer")
        for b in ("author", "graphslim")
    }
    assert len(rows) == len(expected)
    assert {(r["method"], r["dataset"], r["backend"]) for r in rows} == expected
    indexed = {(r["method"], r["dataset"], r["backend"]): r for r in rows}
    if language == "en":
        header = "| Method | Dataset | Author (s) | GraphSlim (s) | Time ratio | Author GCN (%) | GraphSlim GCN (%) |"
    else:
        header = "| 方法 | 数据集 | 作者实现（秒） | GraphSlim（秒） | 耗时比 | 作者实现 GCN（%） | GraphSlim GCN（%） |"
    lines = [header, "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, name in (("cgc", "CGC"), ("clustgdd", "ClustGDD"), ("gcgp", "GCGP")):
        for dataset in ("cora", "citeseer"):
            author = indexed[method, dataset, "author"]
            package = indexed[method, dataset, "graphslim"]
            for row in (author, package):
                assert row["status"] == "complete"
                assert row["completed_seeds"] == row["expected_seeds"] == 3
                assert row["evaluations"] == 30
            assert author["synthetic_nodes"] == package["synthetic_nodes"]
            assert author["device"] == package["device"]
            ratio = author["reduction_seconds_median"] / package["reduction_seconds_median"]
            assert math.isclose(ratio, package["author_over_graphslim_time"])
            values = [
                name,
                dataset.title(),
                f"{author['reduction_seconds_median']:.3f}",
                f"{package['reduction_seconds_median']:.3f}",
                f"{ratio:.2f}×",
                f"{author['accuracy_mean']:.2f} ± {author['accuracy_std']:.2f}",
                f"{package['accuracy_mean']:.2f} ± {package['accuracy_std']:.2f}",
            ]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def replace(text, name, table):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    assert text.count(start) == text.count(end) == 1
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n\n" + table + "\n\n" + end + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    coverage = load_coverage()
    stale = []
    for filename, language in (("README.md", "en"), ("README_zh-CN.md", "zh")):
        path = ROOT / filename
        old = path.read_text(encoding="utf-8")
        new = replace(old, "implementation-comparison", speed_table(language))
        new = replace(new, "reproduction-coverage", coverage_table(coverage, language))
        if args.check:
            if old != new:
                stale.append(filename)
        else:
            path.write_text(new, encoding="utf-8")
    if stale:
        raise SystemExit("Out of date: " + ", ".join(stale))
    print("README highlights match the frozen comparison and coverage evidence.")


if __name__ == "__main__":
    main()

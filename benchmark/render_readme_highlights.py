"""Render the README comparison figure and coverage table from frozen evidence.

Run ``python -m benchmark.render_readme_highlights`` to update both languages,
or append ``--check`` to verify them. The coverage snapshot includes its fixed
roster, formulas, policy credits, reference corrections, and source hashes.
"""

import argparse
import json
import math
from pathlib import Path

from benchmark.render_arxiv_efficiency import speed_figure

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "benchmark/results/reproduction_coverage_20261101.json"


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
        label = "Relative shortfall"
    else:
        header = "| 判定口径 | <1% | <2% | <5% |"
        label = "单边相对短缺"
    lines = [header, "| --- | ---: | ---: | ---: |"]
    cells = []
    for threshold in ("1", "2", "5"):
        row = data["thresholds"]["mean_shortfall_percent"][threshold]
        cells.append(f"{row['count']}/{row['denominator']} ({row['percent']:.2f}%)")
    lines.append("| " + " | ".join([label, *cells]) + " |")
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
        new = replace(old, "implementation-comparison", speed_figure(language))
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

"""Render the README comparison figure and coverage table from frozen evidence.

Run ``python -m benchmark.render_readme_highlights`` to update both languages,
or append ``--check`` to verify them. The coverage snapshot includes its fixed
roster, formulas, policy credits, reference corrections, and source hashes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "benchmark/results/reproduction_coverage_20261101.json"
COMPARISON = ROOT / "benchmark/results/implementation_comparison_20261031.json"
FIGURE = ROOT / "benchmark/results/implementation_comparison_figure.json"


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


def speed_figure(language):
    data = json.loads(FIGURE.read_text(encoding="utf-8"))
    assert data["source_sha256"] == hashlib.sha256(COMPARISON.read_bytes()).hexdigest()
    for suffix, expected in data["assets_sha256"].items():
        asset = ROOT / f"figures/implementation_comparison{suffix}"
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected
    fidelity = data["fidelity"]
    exception = next(p for p in data["pairs"] if (p["method"], p["dataset"]) == ("gcgp", "cora"))
    gap = exception["common_gcn"]["relative_shortfall_percent"]
    if language == "en":
        alt = "Large comparison of reduction time, peak process RSS, peak GPU allocation, and accuracy shortfall"
        note = (
            "**Reproduction criterion: relative accuracy shortfall <5%.** "
            f"**{fidelity['shared_gcn_passes']}/{fidelity['shared_gcn_pairs']}** pairs meet it with shared GCN evaluation; "
            f"GCGP/Cora has **{gap:.2f}%** shortfall. With GCGP's native GP predictor, "
            f"**{fidelity['native_gp_passes']}/{fidelity['native_gp_pairs']}** pairs meet the criterion "
            "(orange diamonds). The threshold concerns predictive quality."
        )
        links = "[Vector PDF](figures/implementation_comparison.pdf) · [High-resolution PNG](figures/implementation_comparison.png) · [Reviewed chart data](benchmark/results/implementation_comparison_figure.json)"
    else:
        alt = "缩减耗时、峰值进程内存、峰值 GPU 显存和准确率相对短缺的大图对比"
        note = (
            "**复现标准：准确率单边相对短缺低于 5%。** "
            f"统一 GCN 评估下 **{fidelity['shared_gcn_passes']}/{fidelity['shared_gcn_pairs']}** 组达标；"
            f"GCGP/Cora 的短缺为 **{gap:.2f}%**。采用 GCGP 原生 GP 预测器时，"
            f"**{fidelity['native_gp_passes']}/{fidelity['native_gp_pairs']}** 组达标（橙色菱形）。"
            "该阈值针对预测效果。"
        )
        links = "[矢量 PDF](figures/implementation_comparison.pdf) · [高清 PNG](figures/implementation_comparison.png) · [经核对的绘图数据](benchmark/results/implementation_comparison_figure.json)"
    image = (
        f"[![{alt}](figures/implementation_comparison.svg)](figures/implementation_comparison.pdf)"
    )
    return "\n\n".join([image, note, links])


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

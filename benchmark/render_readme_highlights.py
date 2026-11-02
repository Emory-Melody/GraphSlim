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

from benchmark.render_arxiv_efficiency import speed_figure

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "benchmark/results/reproduction_coverage_20261101.json"
COVERAGE_FIGURE = ROOT / "benchmark/results/reproduction_coverage_figure.json"


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


def coverage_content(data, language):
    figure = json.loads(COVERAGE_FIGURE.read_text())
    assert figure["source_sha256"] == hashlib.sha256(COVERAGE.read_bytes()).hexdigest()
    assert figure["metric"] == "mean_shortfall_percent"
    assert figure["total_pairs"] == data["total_pairs"]
    for suffix, digest in figure["assets_sha256"].items():
        path = ROOT / f"figures/reproduction_coverage{suffix}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    if language == "en":
        alt = "Cumulative reproduction coverage at 1%, 2% and 5% one-sided relative-shortfall thresholds, with policy credits separately hatched"
        caption = "Each column is cumulative for the same fixed roster; blue segments show disjoint measured-shortfall ranges. The hatched segment contains the same **seven policy-credited Clustering pairs** at every cutoff; their numerical gaps remain unknown."
        links = "[Vector PDF](figures/reproduction_coverage.pdf) · [High-resolution PNG](figures/reproduction_coverage.png) · [Reviewed chart data](benchmark/results/reproduction_coverage_figure.json) · [Figure renderer](benchmark/plot_reproduction_coverage.py)"
    else:
        alt = "单边相对短缺阈值为 1%、2%、5% 时的累计复现覆盖率，规则计入组合单独用纹理标出"
        caption = "各柱按阈值累计同一固定范围内的组合，蓝色区间对应互斥的实测短缺范围。纹理部分为每档均按规则计入的同样 **7 个 Clustering 组合**，其数值差距仍未知。"
        links = "[矢量 PDF](figures/reproduction_coverage.pdf) · [高清 PNG](figures/reproduction_coverage.png) · [经核对的绘图数据](benchmark/results/reproduction_coverage_figure.json) · [绘图脚本](benchmark/plot_reproduction_coverage.py)"
    image = f"[![{alt}](figures/reproduction_coverage.svg)](figures/reproduction_coverage.pdf)"
    return "\n\n".join([image, caption, links, coverage_table(data, language)])


def replace(text, name, table):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    assert text.count(start) == text.count(end) == 1
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n\n" + table + "\n\n" + end + after


def order_highlights(text, language):
    if language == "en":
        acceleration = "## Clustering acceleration\n"
        coverage = "## Reproduction coverage\n"
        old_links = "[Clustering acceleration](#clustering-acceleration) · [Reproduction coverage](#reproduction-coverage)"
        new_links = "[Reproduction coverage](#reproduction-coverage) · [Clustering acceleration](#clustering-acceleration)"
    else:
        acceleration = "## 聚类加速\n"
        coverage = "## 复现覆盖率\n"
        old_links = "[聚类加速](#聚类加速) · [复现覆盖率](#复现覆盖率)"
        new_links = "[复现覆盖率](#复现覆盖率) · [聚类加速](#聚类加速)"
    text = text.replace(old_links, new_links)
    speed_start, coverage_start = text.index(acceleration), text.index(coverage)
    if coverage_start < speed_start:
        return text
    coverage_end = text.find("\n## ", coverage_start + len(coverage))
    if coverage_end < 0:
        coverage_end = len(text)
    else:
        coverage_end += 1
    return (
        text[:speed_start]
        + text[coverage_start:coverage_end]
        + text[speed_start:coverage_start]
        + text[coverage_end:]
    )


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
        new = replace(new, "reproduction-coverage", coverage_content(coverage, language))
        new = order_highlights(new, language)
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

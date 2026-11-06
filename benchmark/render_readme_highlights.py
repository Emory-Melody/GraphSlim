"""Render the README comparison figure and coverage table from frozen evidence.

Run ``python -m benchmark.render_readme_highlights`` to update both languages,
or append ``--check`` to verify them. The coverage snapshot includes its fixed
roster, formulas, reference corrections, exclusions, and source hashes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark.render_arxiv_efficiency import speed_figure
from benchmark.plot_reproduction_coverage import METRIC, THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "benchmark/results/reproduction_coverage_20261106.json"
COVERAGE_FIGURE = ROOT / "benchmark/results/reproduction_coverage_figure.json"


def load_coverage():
    data = json.loads(COVERAGE.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    assert len({(p["method"], p["dataset"]) for p in pairs}) == data["total_pairs"]
    assert sum(p["complete_result"] for p in pairs) == data["complete_pairs"]
    assert not any(p["credited_by_policy"] for p in pairs)
    assert all(
        p["method"] != "clustering" and p["reference_status"] == "numeric_reference" for p in pairs
    )
    for metric, thresholds in data["thresholds"].items():
        for threshold, expected in thresholds.items():
            count = 0
            for pair in pairs:
                value = pair[metric]
                assert math.isfinite(value) and value >= 0
                count += pair["complete_result"] and value < float(threshold)
            assert count == expected["count"]
            assert expected["denominator"] == data["total_pairs"]
            assert math.isclose(expected["percent"], 100 * count / data["total_pairs"])
    return data


def coverage_table(data, language):
    if language == "en":
        header = "| Criterion | " + " | ".join(f"<{t}%" for t in THRESHOLDS) + " |"
        label = "One-sided relative shortfall"
    else:
        header = "| 判定口径 | " + " | ".join(f"<{t}%" for t in THRESHOLDS) + " |"
        label = "单边相对短缺"
    lines = [header, "| --- | " + " | ".join("---:" for _ in THRESHOLDS) + " |"]
    labels = (
        ["One-sided relative shortfall", "Two-sided relative error"]
        if language == "en"
        else ["单边相对短缺", "双边相对误差"]
    )
    for metric, label in zip(["mean_shortfall_percent", "mean_relative_gap_percent"], labels):
        cells = []
        for threshold in THRESHOLDS:
            count = sum(p["complete_result"] and p[metric] < threshold for p in data["pairs"])
            cells.append(
                f"{count}/{data['total_pairs']} ({100 * count / data['total_pairs']:.2f}%)"
            )
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def coverage_content(data, language):
    figure = json.loads(COVERAGE_FIGURE.read_text())
    assert figure["source_sha256"] == hashlib.sha256(COVERAGE.read_bytes()).hexdigest()
    assert figure["metric"] == METRIC
    assert figure["total_pairs"] == data["total_pairs"]
    for suffix, digest in figure["assets_sha256"].items():
        path = ROOT / f"figures/reproduction_coverage{suffix}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    if language == "en":
        alt = f"Cumulative reproduction coverage of {data['total_pairs']} documented-reference pairs at 1%, 2%, 5% and 10% one-sided relative-shortfall thresholds"
        caption = f"Each column is cumulative for the same **{data['total_pairs']} pairs with documented numerical references**; blue segments show disjoint measured-shortfall ranges."
        links = "[Vector PDF](figures/reproduction_coverage.pdf) · [High-resolution PNG](figures/reproduction_coverage.png) · [Reviewed chart data](benchmark/results/reproduction_coverage_figure.json) · [Figure renderer](benchmark/plot_reproduction_coverage.py)"
    else:
        alt = f"{data['total_pairs']} 个有数值参照记录的组合在单边相对短缺阈值 1%、2%、5%、10% 下的累计复现覆盖率"
        caption = f"各柱按阈值累计同样 **{data['total_pairs']} 个有数值参照记录的组合**，蓝色区间对应互斥的实测短缺范围。"
        links = "[矢量 PDF](figures/reproduction_coverage.pdf) · [高清 PNG](figures/reproduction_coverage.png) · [经核对的绘图数据](benchmark/results/reproduction_coverage_figure.json) · [绘图脚本](benchmark/plot_reproduction_coverage.py)"
    image = f"[![{alt}](figures/reproduction_coverage.svg)](figures/reproduction_coverage.pdf)"
    return "\n\n".join([image, caption, links, coverage_table(data, language)])


def coverage_context(text, language, data):
    """Render the coverage explanation and chart from the same selected metric."""

    def rate(threshold):
        pairs = data["pairs"]
        count = sum(p["complete_result"] and p[METRIC] < threshold for p in pairs)
        if language == "en":
            return f"{count}/{len(pairs)} ({100 * count / len(pairs):.2f}%)"
        return f"{count}/{len(pairs)}（{100 * count / len(pairs):.2f}%）"

    if language == "en":
        heading = "## Reproduction coverage\n"
        intro = (
            "The original Yelp pairs retain their historical macro-F1 protocol.\n"
            "The added TSpanner/Yelp pair uses AP matched to the revised GC4NC tables;\n"
            "Other Yelp AP results remain outside this roster."
        )
        summary = (
            f"**{data['complete_pairs']}/{data['total_pairs']} declared method–dataset pairs completed all declared settings.**\n"
            f"**{rate(5)}** meet a mean one-sided relative shortfall below 5%;\n"
            f"**{rate(10)}** meet a mean one-sided relative shortfall below 10%.\n"
            f"The fixed roster spans **{data['method_count']} methods and {len(data['datasets'])} datasets**, with equal weight per pair.\n"
            "Pairs without documented numerical references are excluded."
        )
        formula = (
            "One-sided relative shortfall is `100 * max(0, reference - reproduced) / reference`;\n"
            "a score at or above the reference contributes **0%** shortfall. Compute shortfall per declared\n"
            "matched setting, average within a method–dataset pair, then apply the strict `<` threshold.\n"
            "The displayed 1%, 2%, 5% and 10% cutoffs are relative percentages, not percentage-point gaps."
        )
        boundary = (
            "The denominator is **110 = 90 historical GC4NC pairs + 16 method-paper pairs + 4 updated GC4NC pairs**\n"
            "(method papers: GECC 6, CGC 5, ClustGDD 5). TSpanner adds Arxiv, Flickr, Reddit and Yelp;\n"
            "its existing three pairs are not counted again. Each unique method–dataset pair counts once.\n"
            "Multiple reduction rates, saved edge graphs and evaluation seeds do not add data points.\n"
            "The four added pairs use the same archived observations reported in the revised GC4NC tables;\n"
            "they are documented comparisons rather than independent replication runs.\n"
            "CentP and GDEM are excluded: defects fixed after 1.3.1 also affected the implementation that\n"
            "generated their GC4NC references ([reasons and fresh scores](benchmark/results/reproducibility_20261106/README.md)).\n"
            "The [pair-level snapshot](benchmark/results/reproduction_coverage_20261106.json) records all 110 pairs within the 17 × 7 combinations and their provenance."
        )
    else:
        heading = "## 复现覆盖率\n"
        intro = (
            "原有 Yelp 组合保留历史宏 F1 协议。\n"
            "新增的 TSpanner/Yelp 使用与新版 GC4NC 表格匹配的 AP；\n"
            "其他 Yelp AP 结果不纳入这一范围。"
        )
        summary = (
            f"**固定的 {data['complete_pairs']}/{data['total_pairs']} 个“方法×数据集”组合均已完成全部声明设置。**\n"
            f"**{rate(5)}** 的平均单边相对短缺低于 5%；\n"
            f"**{rate(10)}** 的平均单边相对短缺低于 10%。\n"
            f"该固定范围涉及 **{data['method_count']} 种方法、{len(data['datasets'])} 个数据集**，每个方法与数据集组合等权。\n"
            "没有数值参照记录的组合不计入统计。"
        )
        formula = (
            "单边相对短缺为 `100 * max(0, 参照值 - 复现值) / 参照值`，达到或高于参照值的结果记为 **0%**。\n"
            "先在每个声明且匹配的设置上计算，再在方法与数据集组合内平均，最后按严格 `<` 判断。\n"
            "图中展示 1%、2%、5%、10% 四档相对容差，并非百分点差。"
        )
        boundary = (
            "分母为 **110 = 90 个历史 GC4NC 组合 + 16 个方法原论文组合 + 4 个新版 GC4NC 组合**\n"
            "（方法原论文：GECC 6、CGC 5、ClustGDD 5）。TSpanner 新增 Arxiv、Flickr、Reddit、Yelp 四点，\n"
            "其原有三点不重复计数。每个唯一的“方法×数据集”组合只计一个点，\n"
            "多个缩减率、保存的边稀疏图和评估种子不增加点数。\n"
            "新增四点与新版 GC4NC 表格采用同一批归档观测，是有记录的数值对照，不是独立重跑的复现证据。\n"
            "CentP 和 GDEM 不纳入比较：1.3.1 之后修复的缺陷同样存在于生成其 GC4NC 参照值的实现中\n"
            "（[原因与重跑分数](benchmark/results/reproducibility_20261106/README.md)）。\n"
            "[逐点统计快照](benchmark/results/reproduction_coverage_20261106.json)记录了 17 × 7 范围内这 110 个组合及其来源。"
        )
    if language == "en":
        formula += (
            "\nShortfall measures performance retention for these higher-is-better scores; it does not establish numerical agreement.\n"
            "We therefore also report two-sided relative error, `100 * abs(reference - reproduced) / reference`,\n"
            "on the **same 110 pairs**. For reference 80 and reproduction 88, shortfall is 0% and two-sided error is 10%.\n"
            "Improvements at one setting do not cancel shortfalls at another: clip per setting before averaging.\n"
            "The metric choice was made after historical results existed; it is not a preregistered independent benchmark."
        )
        reconciliation = (
            "Excluding the four reused-observation comparisons leaves 106 numerical comparisons:\n"
            "one-sided <5% **104/106 (98.11%)**, <10% **106/106 (100.00%)**;\n"
            "two-sided <5% **94/106 (88.68%)**, <10% **99/106 (93.40%)**.\n"
            "\n"
            "[Fixed results artifact](benchmark/results/reproducibility_20261106/README.md)\n"
            "contains 313 setting-level reference/reproduced scores, per-pair metrics,\n"
            "source hashes, author-confirmed reference corrections and both metrics.\n"
            "It embeds the unchanged 1.3.1 artifact and recomputes every count without CentP and GDEM.\n"
            "Recompute offline with Python's standard library:\n"
            "\n"
            "```bash\n"
            "python benchmark/results/reproducibility_20261106/verify.py --check\n"
            "```\n"
            "\n"
            "GC4NC is the primary reference under its unified evaluation protocol;\n"
            "the 16 method-paper comparisons retain separate source labels.\n"
            "The three Pubmed corrections are [confirmed by a GC4NC author](docs/gc4nc_reference_corrections.md).\n"
            "Historical runs, provenance replays and validation-selected HPO are combined;\n"
            "this is not one uniform current-checkout run. The artifact reproduces the\n"
            "reported calculations from frozen scores without rerunning training."
        )
    else:
        formula += (
            "\n单边短缺衡量这些越高越好指标的性能保持程度，不能证明数值一致性。\n"
            "因此在**同样的 110 个组合**上并列报告双边相对误差 `100 * abs(参照值 - 复现值) / 参照值`。\n"
            "例如参照值 80、复现值 88 时，短缺为 0%，双边误差为 10%。\n"
            "每个设置先截断再平均，一个设置的提升不会抵消另一个设置的短缺。\n"
            "该指标选择发生在历史结果产生之后，并非预注册的独立基准实验。"
        )
        reconciliation = (
            "剔除 4 个复用观测的比较后，剩余 106 个有数值参照的组合：\n"
            "单边短缺 <5% 为 **104/106（98.11%）**，<10% 为 **106/106（100.00%）**；\n"
            "双边误差 <5% 为 **94/106（88.68%）**，<10% 为 **99/106（93.40%）**。\n"
            "\n"
            "[固定结果产物](benchmark/results/reproducibility_20261106/README.md)\n"
            "保存 313 个设置的参照值和复现值、逐组合指标、源文件哈希、作者确认的参考值更正及两种口径。\n"
            "它内含未改动的 1.3.1 产物，并在去掉 CentP 和 GDEM 后重算全部统计。\n"
            "仅用 Python 标准库即可离线重算：\n"
            "\n"
            "```bash\n"
            "python benchmark/results/reproducibility_20261106/verify.py --check\n"
            "```\n"
            "\n"
            "GC4NC 的统一评价协议是主要参照，另 16 个方法原论文比较保留独立来源标签。\n"
            "三项 Pubmed 更正已由 [GC4NC 作者确认](docs/gc4nc_reference_corrections.md)。\n"
            "统计汇合历史批次、来源对齐重跑和验证集选参，并非当前代码的一次统一全量运行。\n"
            "该产物从固定分数重算统计量，不重新训练。"
        )
    content = (
        "\n\n".join(
            [
                heading.rstrip(),
                intro,
                summary,
                "<!-- reproduction-coverage:start -->",
                coverage_content(data, language),
                "<!-- reproduction-coverage:end -->",
                formula,
                boundary,
                reconciliation,
            ]
        )
        + "\n\n"
    )
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    end = len(text) if end < 0 else end + 1
    return text[:start] + content + text[end:]


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
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Update only reproduction coverage, preserving other README sections.",
    )
    args = parser.parse_args()
    coverage = load_coverage()
    stale = []
    for filename, language in (("README.md", "en"), ("README_zh-CN.md", "zh")):
        path = ROOT / filename
        old = path.read_text(encoding="utf-8")
        new = coverage_context(old, language, coverage)
        if not args.coverage_only:
            new = replace(new, "implementation-comparison", speed_figure(language))
            new = order_highlights(new, language)
        if args.check:
            if old != new:
                stale.append(filename)
        else:
            path.write_text(new, encoding="utf-8")
    if stale:
        raise SystemExit("Out of date: " + ", ".join(stale))
    print(
        "README reproduction coverage matches the reviewed evidence."
        if args.coverage_only
        else "README highlights match the frozen comparison and coverage evidence."
    )


if __name__ == "__main__":
    main()

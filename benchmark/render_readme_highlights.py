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
COVERAGE = ROOT / "benchmark/results/reproduction_coverage_20261102_gc4nc_extended.json"
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
            "The added TSpanner/Yelp and GDEM/Yelp pairs use AP matched to the revised GC4NC tables;\n"
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
            "The denominator is **124 = 100 historical GC4NC pairs + 16 method-paper pairs + 8 updated GC4NC pairs**\n"
            "(method papers: GECC 6, CGC 5, ClustGDD 5). TSpanner and GDEM each add Arxiv, Flickr, Reddit and Yelp;\n"
            "their existing three pairs each are not counted again. Each unique method–dataset pair counts once.\n"
            "Multiple reduction rates, saved edge graphs and evaluation seeds do not add data points.\n"
            "The eight added pairs use the same archived observations reported in the revised GC4NC tables;\n"
            "they are documented comparisons rather than independent replication runs. The\n"
            "[pair-level snapshot](benchmark/results/reproduction_coverage_20261102_gc4nc_extended.json) records all 124 pairs within the 19 × 7 combinations and their provenance."
        )
    else:
        heading = "## 复现覆盖率\n"
        intro = (
            "原有 Yelp 组合保留历史宏 F1 协议。\n"
            "新增的 TSpanner/Yelp 与 GDEM/Yelp 两点使用与新版 GC4NC 表格匹配的 AP；\n"
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
            "分母为 **124 = 100 个历史 GC4NC 组合 + 16 个方法原论文组合 + 8 个新版 GC4NC 组合**\n"
            "（方法原论文：GECC 6、CGC 5、ClustGDD 5）。TSpanner 和 GDEM 各新增 Arxiv、Flickr、Reddit、Yelp 四点，\n"
            "两种方法原有的各三点不重复计数。每个唯一的“方法×数据集”组合只计一个点，\n"
            "多个缩减率、保存的边稀疏图和评估种子不增加点数。\n"
            "新增八点与新版 GC4NC 表格采用同一批归档观测，是有记录的数值对照，不是独立重跑的复现证据。\n"
            "[逐点统计快照](benchmark/results/reproduction_coverage_20261102_gc4nc_extended.json)记录了 19 × 7 范围内这 124 个组合及其来源。"
        )
    if language == "en":
        formula += (
            "\nShortfall measures performance retention for these higher-is-better scores; it does not establish numerical agreement.\n"
            "We therefore also report two-sided relative error, `100 * abs(reference - reproduced) / reference`,\n"
            "on the **same 124 pairs**. For reference 80 and reproduction 88, shortfall is 0% and two-sided error is 10%.\n"
            "Improvements at one setting do not cancel shortfalls at another: clip per setting before averaging.\n"
            "The metric choice was made after historical results existed; it is not a preregistered independent benchmark."
        )
        reconciliation = (
            "The previous public README used **123 pairs / 20 methods**, including seven completed Clustering pairs\n"
            "with unknown numerical gaps that received policy credit at every cutoff. Removing those seven leaves\n"
            "116 numerical comparisons: two-sided <5% **104/116 (89.66%)**, <10% **109/116 (93.97%)**.\n"
            "Adding eight documented GC4NC comparisons gives **123 − 7 + 8 = 124 pairs / 19 methods**;\n"
            "the retained 116 pair records are unchanged. The old 111/123 and 116/123 counts combine measured agreement\n"
            "with completion credits and are preserved only as historical reporting results.\n\n"
            "[Fixed results artifact and full protocol reconciliation](benchmark/results/reproducibility_20261103/README.md)\n"
            "contains 355 setting-level reference/reproduced scores, per-pair metrics, historical rosters,\n"
            "source hashes and both metrics. Recompute offline with Python's standard library:\n\n"
            "```bash\npython benchmark/results/reproducibility_20261103/verify.py --check\n```\n\n"
            "References are versioned GC4NC and GECC/CGC/ClustGDD tables. The artifact discloses three Pubmed\n"
            "corrections, including an **inferred, not author-confirmed** TSpanner cell swap. Historical runs,\n"
            "provenance replays and validation-selected HPO are combined; this is not one uniform current-checkout run.\n"
            "The artifact reproduces the reported calculations from frozen scores, without rerunning training."
        )
    else:
        formula += (
            "\n单边短缺衡量这些越高越好指标的性能保持程度，不能证明数值一致性。\n"
            "因此在**同样的 124 个组合**上并列报告双边相对误差 `100 * abs(参照值 - 复现值) / 参照值`。\n"
            "例如参照值 80、复现值 88 时，短缺为 0%，双边误差为 10%。\n"
            "每个设置先截断再平均，一个设置的提升不会抵消另一个设置的短缺。\n"
            "该指标选择发生在历史结果产生之后，并非预注册的独立基准实验。"
        )
        reconciliation = (
            "先前公开 README 使用 **123 个组合、20 种方法**，包含 7 个已完成但无数值参照的 Clustering 组合，\n"
            "这些组合按历史规则在各阈值获得通过计数。去除这 7 个后，116 个有数值参照组合的双边误差\n"
            "<5% 为 **104/116（89.66%）**，<10% 为 **109/116（93.97%）**。\n"
            "再加入 8 个新版 GC4NC 比较，得到 **123 − 7 + 8 = 124 个组合、19 种方法**；\n"
            "保留的 116 条组合记录未改变。旧版 111/123 和 116/123 混合了数值比较与完成计数，仅作为历史报告保留。\n\n"
            "[固定结果产物与完整协议对照](benchmark/results/reproducibility_20261103/README.md)\n"
            "保存 355 个设置的参照值和复现值、逐组合指标、历史范围、源文件哈希及两种口径。仅用 Python 标准库即可离线重算：\n\n"
            "```bash\npython benchmark/results/reproducibility_20261103/verify.py --check\n```\n\n"
            "参考值固定到 GC4NC 与 GECC/CGC/ClustGDD 的明确版本及表格；产物披露三项 Pubmed 修订，\n"
            "其中 TSpanner 单元交换是**推定修订，未经原作者确认**。统计汇合历史批次、来源对齐重跑和验证集选参，\n"
            "并非当前代码的一次统一实验。该产物重算冻结分数的报告统计，不重新训练模型。"
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

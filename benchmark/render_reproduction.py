"""Freeze the GC4NC Table 9 reproduction and render its README tables.

``collect`` reads completed jobs from the run directories in ``SOURCES`` and writes
``benchmark/results/table9_reproduction.json``. ``render`` writes the README tables
(English and Simplified Chinese) from that snapshot only.
``--check`` verifies that the embedded tables match the snapshot.

Accuracy rows retain the original completed-row, three-percentage-point display rule.
Yelp uses its separate verified AP snapshot; archived macro-F1 is never relabelled.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "benchmark" / "results" / "table9_reproduction.json"
START = "<!-- reproduction-results:start -->"
END = "<!-- reproduction-results:end -->"
THRESHOLD = 3.0

# Later sources replace earlier cells for the listed methods.
SOURCES = [
    dict(run="runs/table9_20261028", methods=None),
    dict(run="runs/table9_gdem_20261029_v2", methods=["gdem"]),
    dict(run="runs/table9_lightweight_20261029", methods=["cgc", "clustgdd"]),
    # Cells GC4NC Table 9 leaves empty ("-"): GraphSlim measurements under the same protocol.
    dict(run="runs/table9_extra_cells_20261101", methods=["t_spanner", "gdem"]),
    # Inductive K-Center and Herding rerun after the embedding-row fix (Flickr, Reddit, Yelp).
    dict(run="runs/rerun_coreset_ind_20261106", methods=["kcenter", "herding"]),
]
DATASETS = ("citeseer", "cora", "pubmed", "ogbn-arxiv", "flickr", "reddit", "yelp")
RATES = {
    "citeseer": ("0.1", "0.25", "0.5"),
    "cora": ("0.1", "0.25", "0.5"),
    "pubmed": ("0.05", "0.1", "0.5"),
    "ogbn-arxiv": ("0.001", "0.005", "0.01"),
    "flickr": ("0.001", "0.005", "0.01"),
    "reddit": ("0.0005", "0.001", "0.002"),
    "yelp": ("0.0005", "0.001", "0.002"),
}
LABELS = {
    "citeseer": ("0.36", "0.90", "1.80"),
    "cora": ("0.50", "1.30", "2.60"),
    "pubmed": ("0.02", "0.03", "0.15"),
    "ogbn-arxiv": ("0.05", "0.25", "0.50"),
    "flickr": ("0.10", "0.50", "1.00"),
    "reddit": ("0.05", "0.10", "0.20"),
    "yelp": ("0.05", "0.10", "0.20"),
}
TITLES = {
    "citeseer": "Citeseer",
    "cora": "Cora",
    "pubmed": "Pubmed",
    "ogbn-arxiv": "Arxiv",
    "flickr": "Flickr",
    "reddit": "Reddit",
    "yelp": "Yelp",
}
METHODS = (
    ("Node sparsification", ("cent_d", "cent_p", "random", "herding", "kcenter")),
    ("Edge sparsification", ("t_spanner",)),
    ("Coarsening", ("averaging", "clustering", "variation_neighborhoods", "vng")),
    (
        "Condensation",
        (
            "geom",
            "sfgc",
            "gcsntk",
            "gcdmx",
            "doscondx",
            "gcond",
            "doscond",
            "msgc",
            "sgdd",
            "gdem",
            "gecc",
            "cgc",
            "clustgdd",
        ),
    ),
)
NAMES = {
    "cent_d": "Cent-D",
    "cent_p": "Cent-P",
    "random": "Random",
    "herding": "Herding",
    "kcenter": "K-Center",
    "t_spanner": "TSpanner",
    "averaging": "Averaging",
    "clustering": "Clustering",
    "variation_neighborhoods": "VN",
    "vng": "VNG",
    "geom": "GEOM",
    "sfgc": "SFGC",
    "gcsntk": "GCSNTK",
    "gcdmx": "GCDMX",
    "doscondx": "DosCondX",
    "gcond": "GCond",
    "doscond": "DosCond",
    "msgc": "MSGC",
    "sgdd": "SGDD",
    "gdem": "GDEM",
    "gecc": "GECC",
    "cgc": "CGC",
    "clustgdd": "ClustGDD",
    "whole": "Whole",
}
ZH = {
    "Node sparsification": "节点稀疏化",
    "Edge sparsification": "边稀疏化",
    "Coarsening": "粗化",
    "Condensation": "凝聚",
}
# GECC reports one non-evolving ratio per dataset (arXiv:2502.17614v3, Table 9).
GECC_PAPER = {
    ("citeseer", "0.5"): 73.25,
    ("cora", "0.5"): 82.99,
    ("pubmed", "0.5"): 80.24,
    ("flickr", "0.01"): 46.63,
    ("ogbn-arxiv", "0.01"): 66.71,
    ("reddit", "0.001"): 91.37,
}

# CGC, WWW 2025 (arXiv:2405.13707v2), Table 3, structured CGC column.
CGC_PAPER = {
    ("cora", "0.25"): 82.7,
    ("cora", "0.5"): 82.3,
    ("citeseer", "0.25"): 72.5,
    ("citeseer", "0.5"): 72.4,
    ("ogbn-arxiv", "0.001"): 64.1,
    ("ogbn-arxiv", "0.005"): 66.4,
    ("ogbn-arxiv", "0.01"): 67.2,
    ("flickr", "0.001"): 46.8,
    ("flickr", "0.005"): 47.1,
    ("flickr", "0.01"): 47.0,
    ("reddit", "0.0005"): 90.6,
    ("reddit", "0.001"): 91.4,
    ("reddit", "0.002"): 91.6,
}
# ClustGDD, KDD 2025 (arXiv:2505.20807v1), Table 3, structured ClustGDD column.
CLUSTGDD_PAPER = {
    ("cora", "0.25"): 80.8,
    ("cora", "0.5"): 81.7,
    ("citeseer", "0.25"): 73.5,
    ("citeseer", "0.5"): 74.4,
    ("ogbn-arxiv", "0.001"): 63.7,
    ("ogbn-arxiv", "0.005"): 67.5,
    ("ogbn-arxiv", "0.01"): 67.7,
    ("flickr", "0.001"): 46.9,
    ("flickr", "0.005"): 46.9,
    ("flickr", "0.01"): 47.1,
    ("reddit", "0.0005"): 89.8,
    ("reddit", "0.001"): 91.5,
}
OWN_PAPER = {"gecc": GECC_PAPER, "cgc": CGC_PAPER, "clustgdd": CLUSTGDD_PAPER}
# GC4NC took these scores from the authors' implementations. Their expert-trajectory
# training is too slow to rerun for the full matrix, so the README shows the
# published values for these rows.
PUBLISHED_ONLY = ("geom", "sfgc")
# Defects fixed after 1.3.1 also affected the code behind these GC4NC references,
# so their rows are not compared with published values.
EXCLUDED_FROM_REFERENCE = ("cent_p", "gdem")
TABLE9_METHODS = tuple(m for _, ms in METHODS for m in ms if m not in OWN_PAPER) + ("whole",)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def published():
    from benchmark.reproduce_table9 import CSV_NAMES, published as table9

    rows = table9()

    def text(dataset, rate, method):
        label = LABELS[dataset][0 if rate is None else RATES[dataset].index(rate)]
        reference = "GCondX" if method == "gcondx" else NAMES[method]
        return rows[(CSV_NAMES[dataset], label)].get(reference, "")

    def value(dataset, rate, method):
        if method in OWN_PAPER:
            return OWN_PAPER[method].get((dataset, rate))
        try:
            return float(text(dataset, rate, method).split("±")[0])
        except ValueError:
            return None

    value.text = text
    return value


def collect():
    paper = published()
    cells = {}
    for source in SOURCES:
        folder = ROOT / source["run"]
        manifest = read_json(folder / "manifest.json")
        groups = {}
        for job in manifest["jobs"]:
            implementation = (
                manifest.get("replay", {}).get(job["method"], {}).get("method", job["method"])
            )
            if implementation not in NAMES:
                continue
            if source["methods"] is None or job["method"] in source["methods"]:
                groups.setdefault((job["dataset"], job["method"], job["rate"]), []).append(job)
        for (dataset, method, rate), jobs in groups.items():
            paths = [folder / "jobs" / job["id"] / "result.json" for job in jobs]
            errors = [folder / "jobs" / job["id"] / "error.json" for job in jobs]
            cell = dict(
                dataset=dataset,
                method=method,
                rate=rate,
                label=None if rate is None else LABELS[dataset][RATES[dataset].index(rate)],
                paper=paper(dataset, rate, method),
                paper_text=paper.text(dataset, rate, method)
                if method in TABLE9_METHODS or method == "gcondx"
                else "",
                run=source["run"],
                source_sha256=manifest["source_sha256"],
            )
            if all(path.exists() for path in paths):
                records = [read_json(path) for path in paths]
                scores = [100 * e["test"] for r in records for e in r["evaluations"]]
                cell.update(
                    status="complete",
                    mean=round(statistics.mean(scores), 4),
                    std=round(statistics.pstdev(scores), 4),
                    n=len(scores),
                    reduced_nodes=[r.get("reduced_nodes") for r in records],
                    seconds=round(sum(r["elapsed_seconds"] for r in records), 1),
                    result_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                )
            elif any(path.exists() for path in errors):
                kinds = sorted(
                    {read_json(path).get("kind", "ERROR") for path in errors if path.exists()}
                )
                cell.update(status="failed", kind="OOM" if "OOM" in kinds else kinds[0])
            else:
                cell.update(status="incomplete")
            cells[(dataset, method, rate)] = cell
    snapshot = dict(
        sources=SOURCES,
        threshold=THRESHOLD,
        protocol=read_json(ROOT / SOURCES[0]["run"] / "manifest.json")["protocol"],
        replay=read_json(ROOT / SOURCES[0]["run"] / "manifest.json").get("replay", {}),
        cells=sorted(
            cells.values(),
            key=lambda c: (DATASETS.index(c["dataset"]), c["method"], c["rate"] or ""),
        ),
    )
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def index(snapshot):
    """Display archived rows by verified implementation, retaining settings provenance."""
    result = {}
    replay = snapshot.get("replay", {})
    for cell in snapshot["cells"]:
        method = cell["method"]
        mapping = replay.get(method, {})
        implementation = mapping.get("method", method)
        if implementation not in NAMES:
            continue
        if implementation != method:
            cell = dict(
                cell,
                method=implementation,
                legacy_method=method,
                legacy_config=mapping.get("config"),
            )
        key = (cell["dataset"], implementation, cell["rate"])
        if key not in result or method == implementation:
            result[key] = cell
    return result


def row_status(cells, dataset, method):
    """Return (status, mean absolute gap) for one dataset row."""
    row = [cells.get((dataset, method, rate)) for rate in RATES[dataset]]
    if any(cell is None for cell in row):
        return "not run", None
    if any(cell["status"] != "complete" for cell in row):
        failed = [cell for cell in row if cell["status"] == "failed"]
        return ("failed: " + failed[0]["kind"]) if failed else "incomplete", None
    gaps = [abs(cell["mean"] - cell["paper"]) for cell in row if cell["paper"] is not None]
    if not gaps:
        return "no published value", None
    gap = statistics.mean(gaps)
    return ("reproduced" if gap <= THRESHOLD else "outside limit"), gap


def fmt(cell):
    return f"{cell['mean']:.2f} ± {cell['std']:.2f}"


def readme_yelp_ap(snapshot, language="en"):
    """Render AP measurements without substituting historical macro-F1 cells."""
    cells = {(c["method"], c["rate"]): c for c in snapshot.get("cells", [])}

    def values(method):
        row = [cells.get((method, rate)) for rate in RATES["yelp"]]
        if any(c is None or c["status"] != "complete" for c in row):
            return ["NR"] * 3
        if any(c.get("metric") != "AP" for c in row):
            raise ValueError(f"Non-AP measurement in Yelp AP table: {method}")
        return [fmt(c) for c in row]

    title = "#### Yelp — AP (%)" if language == "en" else "#### Yelp — AP（%）"
    header = ["Family", "Method"] if language == "en" else ["类别", "方法"]
    lines = [
        title,
        "",
        "| " + " | ".join(header + [f"{r}%" for r in LABELS["yelp"]]) + " |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family, methods in METHODS:
        if family == "Edge sparsification":
            continue
        for method in methods:
            legacy = any(cells.get((method, r), {}).get("legacy_method") for r in RATES["yelp"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        family if language == "en" else ZH[family],
                        NAMES[method] + ("¹" if legacy else ""),
                        *values(method),
                    ]
                )
                + " |"
            )
    whole = cells.get(("whole", None))
    if whole and whole["status"] == "complete":
        if whole.get("metric") != "AP":
            raise ValueError("Non-AP Whole baseline in Yelp AP table")
        label = "Whole graph" if language == "en" else "全图训练"
        lines += ["", f"{label}: {fmt(whole)}."]
    if language == "en":
        lines += [
            "",
            "TSpanner retains all training nodes; Graphs 1–3 below are saved edge-sparsified graphs, not node-budget levels.",
            "",
            "| Edge method | Graph 1 AP (%) | Graph 2 AP (%) | Graph 3 AP (%) |",
        ]
    else:
        lines += [
            "",
            "TSpanner 保留全部训练节点；下列 Graph 1–3 为已保存的边稀疏图，不对应节点预算。",
            "",
            "| 边稀疏化方法 | Graph 1 AP（%） | Graph 2 AP（%） | Graph 3 AP（%） |",
        ]
    lines += [
        "| --- | ---: | ---: | ---: |",
        "| " + " | ".join(["TSpanner", *values("t_spanner")]) + " |",
        "",
    ]
    if language == "en":
        lines += [
            "AP ranks anomaly label 1 by continuous log-odds. All available rows and Whole use the same twelve-candidate GCN search: three validation seeds select the configuration and validation AP selects the epoch over 300 epochs. Each saved graph receives ten final fits; Random, Herding and K-Center pool three reduction seeds (30 fits). Values are mean ± population standard deviation. Saved graph generation and historical checkpoint selection remain fixed. NR means no complete AP result in this snapshot; historical macro-F1 scores are not substituted. The constant-score AP is 14.53%."
        ]
    else:
        lines += [
            "AP 使用连续对数几率分数对异常类（标签 1）排序。所有可用行及 Whole 共用 12 组 GCN 候选配置：以 3 个验证种子的平均 AP 选配置，并在 300 轮中按验证 AP 选训练检查点。每个已保存图评估 10 次；Random、Herding 和 K-Center 汇总 3 个缩减种子，共 30 次。数值为均值 ± 总体标准差。图生成及其历史检查点选择保持固定。NR 表示此快照中暂无完整 AP 结果，不以历史宏 F1 分数替代。常数分数的 AP 为 14.53%。"
        ]
    source_label = "Frozen AP results and protocol" if language == "en" else "冻结的 AP 结果与协议"
    lines += [f"[{source_label}](benchmark/results/yelp_ap_20261101.json).", ""]
    return "\n".join(lines)


def readme_tables(snapshot, language="en"):
    """Keep the original accuracy-row selection and label each displayed source."""
    cells = index(snapshot)
    lines = []
    for dataset in DATASETS:
        if dataset == "yelp":
            ap_path = ROOT / "benchmark/results/yelp_ap_20261101.json"
            lines += [readme_yelp_ap(read_json(ap_path) if ap_path.exists() else {}, language)]
            continue
        title = (
            f"#### {TITLES[dataset]} — test accuracy (%)"
            if language == "en"
            else f"#### {TITLES[dataset]} — 测试准确率（%）"
        )
        header = (
            ["Family", "Method", "Source / status"]
            if language == "en"
            else ["类别", "方法", "来源 / 状态"]
        )
        rows, edge_rows = [], []
        for family, methods in METHODS:
            for method in methods:
                if method in EXCLUDED_FROM_REFERENCE:
                    continue
                row = [cells.get((dataset, method, rate)) for rate in RATES[dataset]]
                if not any(row):
                    continue
                if method in PUBLISHED_ONLY:
                    if not all(
                        c and c.get("paper_text") not in (None, "", "-", "OOM", "NR") for c in row
                    ):
                        continue
                    values = [
                        (c.get("paper_text") or "NR").replace("±", " ± ") if c else "NR"
                        for c in row
                    ]
                    source = "Published / SE" if language == "en" else "文献数值 / SE"
                    name = NAMES[method] + "²"
                else:
                    status, _ = row_status(cells, dataset, method)
                    if status not in ("reproduced", "no published value"):
                        continue
                    values = [fmt(c) for c in row]
                    source = "Archived run / SD" if language == "en" else "历史运行 / SD"
                    legacy = any(c and c.get("legacy_method") for c in row)
                    name = NAMES[method] + ("¹" if legacy else "")
                if family == "Edge sparsification":
                    edge_rows.append([name, source, *values])
                else:
                    rows.append([family if language == "en" else ZH[family], name, source, *values])
        lines += [
            title,
            "",
            "| " + " | ".join(header + [f"{label}%" for label in LABELS[dataset]]) + " |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        if edge_rows:
            edge_header = (
                ["Edge method", "Source / status", "Graph 1", "Graph 2", "Graph 3"]
                if language == "en"
                else ["边方法", "来源 / 状态", "Graph 1", "Graph 2", "Graph 3"]
            )
            lines += [
                "",
                "Saved edge graphs retain the input nodes; Graph 1–3 are not node-budget levels."
                if language == "en"
                else "已保存的边稀疏图保留输入节点；Graph 1–3 不表示节点预算。",
                "",
                "| " + " | ".join(edge_header) + " |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            lines += ["| " + " | ".join(row) + " |" for row in edge_rows]
        whole = cells.get((dataset, "whole", None))
        if whole and whole["status"] == "complete":
            label = (
                "Whole graph (archived run / SD)"
                if language == "en"
                else "全图训练（历史运行 / SD）"
            )
            lines += ["", f"{label}: {fmt(whole)}."]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", []


def readme_section(snapshot, language="en"):
    tables, _ = readme_tables(snapshot, language)
    if language == "en":
        note = (
            "¹ Archived DosCondX measurements retain their original settings and source names; the Reddit configuration differs from current defaults.\n"
            "² Published GC4NC values use mean ± standard error (SE). These are reference values from the authors' GEOM/SFGC implementations, not package reruns.\n"
            "Other numeric archived runs use mean ± population standard deviation (SD). Accuracy rows with published references are shown when all declared rates completed and their mean absolute gap is at most 3 percentage points.\n"
            "CentP and GDEM are excluded from historical reference comparisons because defects also affected the reference implementations ([reasons and fresh scores](benchmark/results/reproducibility_20261106/README.md)). Their separately archived Yelp AP scores above describe the saved graphs generated at that time, before the fixes.\n"
            "This table is one archived campaign, separate from the 110-pair cumulative reproduction roster. The [pair-level artifact](benchmark/results/reproducibility_20261106/pairs.csv) records the adopted source per pair."
        )
    else:
        note = (
            "¹ DosCondX 历史测量保留原设置和来源名；Reddit 配置与当前默认值不同。\n"
            "² GC4NC 文献值为均值 ± 标准误（SE），来自原作者的 GEOM/SFGC 实现，不属于本包重跑。\n"
            "其他历史运行数值为均值 ± 总体标准差（SD）。有论文参考值的准确率行仅在全部声明比例完成、平均绝对差不超过 3 个百分点时展示。\n"
            "CentP 和 GDEM 不参与历史参考比较，因为缺陷也存在于参考实现中（[原因及重跑结果](benchmark/results/reproducibility_20261106/README.md)）。上方单独归档的 Yelp AP 仍描述当时生成的图，生成时间早于修复。\n"
            "此表对应一个历史批次，与累计的 110 个复现组合分别报告；各组合采用的来源见[逐对产物](benchmark/results/reproducibility_20261106/pairs.csv)。"
        )
    return tables + "\n" + note + "\n"


def embed(path, section):
    text = path.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(f"{path.name}: reproduction markers missing or duplicated")
    before, rest = text.split(START)
    _, after = rest.split(END)
    return before + START + "\n\n" + section + "\n" + END + after


def targets(snapshot):
    return {
        ROOT / "README.md": lambda: embed(ROOT / "README.md", readme_section(snapshot, "en")),
        ROOT / "README_zh-CN.md": lambda: embed(
            ROOT / "README_zh-CN.md", readme_section(snapshot, "zh")
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("collect", "render"))
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args(argv)
    if options.action == "collect":
        collect()
    snapshot = read_json(SNAPSHOT)
    if options.check:
        stale = [
            path.name
            for path, make in targets(snapshot).items()
            if path.read_text(encoding="utf-8") != make()
        ]
        if stale:
            raise SystemExit(f"Out of date: {', '.join(stale)}")
        print("READMEs match the reproduction snapshot.")
    elif options.action in ("collect", "render"):
        for path, make in targets(snapshot).items():
            path.write_text(make(), encoding="utf-8")


if __name__ == "__main__":
    main()

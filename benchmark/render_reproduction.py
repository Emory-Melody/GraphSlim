"""Freeze the GC4NC Table 9 reproduction and render its README tables.

``collect`` reads completed jobs from the run directories in ``SOURCES`` and writes
``benchmark/results/table9_reproduction.json``. ``render`` writes the README tables
(English and Simplified Chinese) from that snapshot only.
``--check`` verifies that the embedded tables match the snapshot.

A dataset row appears in the README when all three rates completed and the mean
absolute difference from the published value is at most ``THRESHOLD`` points.
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
    ("Coarsening", ("averaging", "variation_neighborhoods", "vng")),
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


def readme_tables(snapshot, language="en"):
    cells = index(snapshot)
    lines = []
    missing = []
    for dataset in DATASETS:
        if language == "en":
            metric = "F1-macro" if dataset == "yelp" else "test accuracy"
            title = f"#### {TITLES[dataset]} — {metric} (%)"
            header = ["Family", "Method"]
        else:
            metric = "宏平均 F1" if dataset == "yelp" else "测试准确率"
            title = f"#### {TITLES[dataset]} — {metric}（%）"
            header = ["类别", "方法"]
        rows = []
        for family, methods in METHODS:
            for method in methods:
                if method in PUBLISHED_ONLY:
                    texts = [
                        cells[(dataset, method, rate)]["paper_text"]
                        for rate in RATES[dataset]
                        if (dataset, method, rate) in cells
                    ]
                    if len(texts) == 3 and all(t not in ("", "-", "OOM") for t in texts):
                        values = [t.replace("±", " ± ") for t in texts]
                        rows.append(
                            [
                                family if language == "en" else ZH[family],
                                NAMES[method] + "²",
                                *values,
                            ]
                        )
                    continue
                status, _ = row_status(cells, dataset, method)
                if status == "no published value":
                    values = [fmt(cells[(dataset, method, rate)]) for rate in RATES[dataset]]
                    rows.append(
                        [family if language == "en" else ZH[family], NAMES[method] + "³", *values]
                    )
                    continue
                if status != "reproduced":
                    if (dataset, method, RATES[dataset][0]) in cells:
                        missing.append((method, dataset, status))
                    continue
                values = [fmt(cells[(dataset, method, rate)]) for rate in RATES[dataset]]
                legacy = any(
                    cells[(dataset, method, rate)].get("legacy_method") for rate in RATES[dataset]
                )
                name = NAMES[method] + ("¹" if legacy else "")
                rows.append([family if language == "en" else ZH[family], name, *values])
        whole = cells.get((dataset, "whole", None))
        lines += [title, ""]
        lines.append("| " + " | ".join(header + [f"{label}%" for label in LABELS[dataset]]) + " |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        if whole and whole["status"] == "complete":
            label = "Whole graph" if language == "en" else "全图训练"
            lines += ["", f"{label}: {fmt(whole)}."]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", missing


def readme_section(snapshot, language="en"):
    tables, missing = readme_tables(snapshot, language)

    def listed(selected):
        names = {}
        for method, dataset, _ in selected:
            names.setdefault(NAMES[method], []).append(TITLES[dataset])
        return "; ".join(f"{m} ({', '.join(d)})" for m, d in names.items())

    outside = listed([m for m in missing if m[2] == "outside limit"])
    failed = listed([m for m in missing if m[2].startswith("failed")])
    running = listed([m for m in missing if m[2] in ("incomplete", "not run")])
    if language == "en":
        note = (
            "¹ Archived DosCondX measurements use the archived experimental settings and "
            "corresponding reference row. The Reddit configuration differs from current DosCondX defaults; "
            "raw archived names and configurations are preserved.\n"
            "² Published GC4NC values (mean ± standard error). The paper obtained GEOM and "
            "SFGC scores from the authors' implementations; their expert-trajectory "
            "training is too slow to rerun for the full matrix.\n"
            "³ The published GC4NC benchmark reports no value for these cells; "
            "GraphSlim measurements "
            "under the same protocol are shown without a published comparison."
        )
        extra = []
        if outside:
            extra.append(
                f"Rows more than {THRESHOLD:.0f} points from the paper on average are omitted: {outside}."
            )
        if failed:
            extra.append(f"Runs that failed: {failed}.")
        if running:
            extra.append(f"Still running: {running}.")
    else:
        note = (
            "¹ 这些历史测量实际运行 DosCondX，采用归档实验配置及对应的论文参考行；其中 Reddit 配置与当前 DosCondX 默认值不同，原始记录和配置均保留。\n"
            "² GC4NC 论文数值（均值 ± 标准误）。论文中 GEOM 和 SFGC 的得分来自原作者实现，"
            "其专家轨迹训练太慢，未对完整矩阵重跑。\n"
            "³ GC4NC 论文未给出这些单元的数值；此处为相同协议下的 GraphSlim 测量，无论文对照。"
        )
        extra = []
        if outside:
            extra.append(f"与论文平均相差超过 {THRESHOLD:.0f} 个百分点的行未列出：{outside}。")
        if failed:
            extra.append(f"运行失败：{failed}。")
        if running:
            extra.append(f"仍在运行：{running}。")
    parts = [tables.rstrip(), "", note]
    if extra:
        parts += ["", " ".join(extra)]
    return "\n".join(parts) + "\n"


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

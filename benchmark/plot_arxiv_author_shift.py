"""Plot published author points and frozen GraphSlim ogbn-arxiv measurements.

Run from the repository root: python -m benchmark.plot_arxiv_author_shift
The arrows compare reported author experiments with local package runs; they are
descriptive because hardware and evaluation settings differ between studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "benchmark/results/tradeoff_20261028.json"
OUTPUT = ROOT / "benchmark/results/arxiv_author_vs_graphslim_20261029.json"

CGC_PAPER = "https://arxiv.org/html/2405.13707"
CLUSTGDD_PAPER = "https://arxiv.org/html/2505.20807"
GCGP_PAPER = "https://arxiv.org/html/2501.02565"

# CGC paper, Table 3 (test accuracy) and Table 4 (condensation time), Arxiv
# r=0.25%. The author code maps that rate to 0.005 of the labeled training set.
PAPER_LANDSCAPE = {
    "VN": (412.6, 43.5),
    "A-CM": (179.9, 60.0),
    "GCond": (14876.0, 63.2),
    "GCond-X": (13516.7, 64.2),
    "GCDM": (690.0, 59.6),
    "GCDM-X": (469.0, 61.2),
    "DosCond": (12483.0, 63.5),
    "SGDD": (22155.8, 65.8),
    "MSGC": (14723.1, 65.8),
    "SNTK": (9353.6, 65.1),
    "SNTK-X": (9079.9, 65.5),
    "SimGC": (252.4, 66.4),
    "SFGC": (86288.5, 66.1),
    "GEOM": (108962.5, 67.5),
    "GCSR": (5283.5, 65.4),
    "CGC": (8.8, 66.4),
    "CGC-X": (7.9, 66.3),
}

PAIRS = (
    {
        "method": "CGC",
        "rate": "0.005",
        "nodes": 454,
        "paper_seconds": 8.8,
        "paper_accuracy_percent": 66.4,
        "paper_time_approximate": False,
        "paper": CGC_PAPER,
        "paper_location": "Tables 3 and 4, Arxiv r=0.25%",
        "color": "#E86B3A",
    },
    {
        "method": "ClustGDD",
        "rate": "0.001",
        "nodes": 90,
        "paper_seconds": 79.0,
        "paper_accuracy_percent": 63.7,
        "paper_time_approximate": True,
        "paper": CLUSTGDD_PAPER,
        "paper_location": "Table 3, arXiv r=0.05%; Appendix D.1 says around 79 s",
        "color": "#3C70E8",
    },
    {
        "method": "GCGP",
        "rate": "0.01",
        "nodes": 909,
        "paper_seconds": 298.4082,
        "paper_accuracy_percent": 65.7,
        "paper_time_approximate": False,
        "paper": GCGP_PAPER,
        "paper_location": "Tables III and IV, Ogbn-arxiv size 909, X-only variant",
        "color": "#20A987",
    },
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    snapshot = json.loads(SNAPSHOT.read_text())
    assert snapshot["source"]["source_sha256"] == (
        "66e175c052a27271d5dbda4adb1d587e02fa9ae69c3171d7c78f3f315481dda0"
    )
    run_dir = ROOT / snapshot["source"]["run_dir"]
    assert sha(run_dir / "source.zip") == snapshot["source"]["archive_sha256"]
    assert sha(run_dir / "manifest.json") == snapshot["source"]["manifest_sha256"]
    pairs = []
    for item in PAIRS:
        row = next(
            cell
            for cell in snapshot["cells"]
            if cell["dataset"] == "ogbn-arxiv"
            and cell["method"] == item["method"].lower()
            and cell["training_node_rate"] == item["rate"]
        )
        assert row["n_evaluations"] == 30
        assert row["actual_nodes"] == [item["nodes"]] * 3
        for job_id, digest in zip(row["job_ids"], row["result_sha256"]):
            assert sha(run_dir / "jobs" / job_id / "result.json") == digest
        pairs.append(
            {key: value for key, value in item.items() if key not in ("color", "rate")}
            | {
                "graphslim_training_node_rate": item["rate"],
                "graphslim_seconds": row["median_reduction_seconds"],
                "graphslim_accuracy_percent": 100 * row["test_accuracy_mean"],
                "graphslim_evaluations": row["n_evaluations"],
                "graphslim_job_ids": row["job_ids"],
                "graphslim_result_sha256": row["result_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "scope": "Published author reports to frozen GraphSlim package measurements on ogbn-arxiv",
        "comparison_type": "descriptive cross-study arrows; different hardware and evaluation settings",
        "x_axis": "graph condensation/reduction time in seconds, log scale",
        "y_axis": "test accuracy percentage",
        "uncertainty_displayed": False,
        "paper_landscape": [
            {"method": name, "seconds": seconds, "accuracy_percent": accuracy}
            for name, (seconds, accuracy) in PAPER_LANDSCAPE.items()
        ],
        "paper_landscape_source": CGC_PAPER,
        "paper_landscape_location": "Tables 3 and 4, Arxiv r=0.25%",
        "graphslim_snapshot": str(SNAPSHOT.relative_to(ROOT)),
        "graphslim_snapshot_sha256": sha(SNAPSHOT),
        "graphslim_source": snapshot["source"],
        "graphslim_hardware": snapshot["hardware"],
        "graphslim_protocol": snapshot["protocol"],
        "pairs": pairs,
    }


def validate(data: dict) -> None:
    """Check the portable plot data against the checked-in measurement snapshot."""
    snapshot = json.loads(SNAPSHOT.read_text())
    assert data["graphslim_snapshot_sha256"] == sha(SNAPSHOT)
    assert data["paper_landscape"] == [
        {"method": name, "seconds": seconds, "accuracy_percent": accuracy}
        for name, (seconds, accuracy) in PAPER_LANDSCAPE.items()
    ]
    assert len(data["pairs"]) == len(PAIRS)
    for pair, published in zip(data["pairs"], PAIRS):
        for key, value in published.items():
            if key not in ("color", "rate"):
                assert pair[key] == value
        row = next(
            cell
            for cell in snapshot["cells"]
            if cell["dataset"] == "ogbn-arxiv"
            and cell["method"] == pair["method"].lower()
            and cell["training_node_rate"] == published["rate"]
        )
        assert row["actual_nodes"] == [pair["nodes"]] * 3
        assert pair["graphslim_evaluations"] == row["n_evaluations"] == 30
        assert pair["graphslim_seconds"] == row["median_reduction_seconds"]
        assert pair["graphslim_accuracy_percent"] == 100 * row["test_accuracy_mean"]
        assert pair["graphslim_result_sha256"] == row["result_sha256"]


def render(data: dict) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "graphslim-arxiv-author-shift",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12.8, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.20, top=0.79)
    fig.text(0.085, 0.94, "ogbn-arxiv: accuracy–reduction time", size=21, weight="bold")
    fig.text(0.085, 0.885, "Author paper  →  GraphSlim package run", size=13)
    ax.set_xscale("log")
    ax.set_xlim(1.5, 160000)
    ax.set_ylim(0, 78)
    ax.set_xlabel("Graph reduction time (seconds, log scale)", weight="bold")
    ax.set_ylabel("Test accuracy (%)", weight="bold")
    ax.grid(True, color="#E8E8E8", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C4C4C4")
    ax.tick_params(length=0, pad=8)

    landscape = data["paper_landscape"]
    ax.scatter(
        [p["seconds"] for p in landscape],
        [p["accuracy_percent"] for p in landscape],
        s=42,
        color="#B3B6BE",
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
    )
    for method, dx, dy in (("GCond", 8, -14), ("GEOM", -46, 7)):
        item = next(p for p in landscape if p["method"] == method)
        ax.annotate(
            method,
            (item["seconds"], item["accuracy_percent"]),
            xytext=(dx, dy),
            textcoords="offset points",
            color="#777D88",
            size=9,
        )

    colors = {item["method"]: item["color"] for item in PAIRS}
    for row in data["pairs"]:
        name = row["method"]
        color = colors[name]
        x0, y0 = row["paper_seconds"], row["paper_accuracy_percent"]
        x1, y1 = row["graphslim_seconds"], row["graphslim_accuracy_percent"]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.6, "mutation_scale": 17},
            zorder=4,
        )
        ax.scatter(
            [x0], [y0], s=130, facecolors="white", edgecolors=color, linewidths=2.2, zorder=5
        )
        ax.scatter([x1], [y1], s=140, color=color, edgecolors="white", linewidths=1.2, zorder=6)

    # Labels sit next to the pair endpoints and state the exact reduced-node budget.
    for name, ox, oy, gx, gy in (
        ("CGC · 454 nodes", -8, 16, 8, -18),
        ("ClustGDD · 90 nodes", 12, -25, 8, -20),
        ("GCGP · 909 nodes", 8, 19, 8, 8),
    ):
        row = next(p for p in data["pairs"] if p["method"] == name.split(" · ")[0])
        color = colors[row["method"]]
        ax.annotate(
            name,
            (row["paper_seconds"], row["paper_accuracy_percent"]),
            xytext=(ox, oy),
            textcoords="offset points",
            color=color,
            size=10,
            weight="bold",
            ha="right" if ox < 0 else "left",
        )
        ax.annotate(
            "GraphSlim",
            (row["graphslim_seconds"], row["graphslim_accuracy_percent"]),
            xytext=(gx, gy),
            textcoords="offset points",
            color=color,
            size=9,
            weight="bold",
        )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#B3B6BE",
            markeredgecolor="white",
            markersize=8,
            label="CGC paper context (17 methods)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=1.7,
            markersize=9,
            label="Author paper",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=9,
            label="GraphSlim",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#E0E0E0",
        fontsize=9,
    )
    fig.text(
        0.085,
        0.105,
        "Arrows match reduced-node count. ClustGDD paper time is approximately 79 s.",
        size=10,
    )
    fig.text(
        0.085,
        0.065,
        "Paper runs and GraphSlim runs used different hardware and evaluation settings; arrows show reported positions.",
        size=10,
        color="#555555",
    )
    paths = []
    for extension in ("svg", "png", "pdf"):
        path = ROOT / "figures" / f"arxiv_author_vs_graphslim.{extension}"
        metadata = (
            {"Date": None}
            if extension == "svg"
            else {"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {}
        )
        fig.savefig(path, dpi=200, facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect", action="store_true", help="also verify local raw run files and source ZIP"
    )
    options = parser.parse_args()
    data = collect() if options.collect else json.loads(OUTPUT.read_text())
    validate(data)
    paths = render(data)
    data["figure_sha256"] = {path.name: sha(path) for path in paths}
    data["plot_script_sha256"] = sha(Path(__file__))
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"Validated {len(data['paper_landscape'])} paper context points and {len(data['pairs'])} arrows"
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Plot the verified Products snapshot with blue/gray, hatched bar charts.

Run: python -m benchmark.plot_products
Requires matplotlib; reads recorded results and never launches training.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("clustgdd", "cgc", "gcgp", "random", "whole")
NAMES = {
    "clustgdd": "ClustGDD",
    "cgc": "CGC",
    "gcgp": "GCGP",
    "random": "Random",
    "whole": "Full graph",
}
COLORS = {
    "clustgdd": "#4F6BFF",
    "cgc": "#A8BDFF",
    "gcgp": "#D6E3FF",
    "random": "#DEDEDE",
    "whole": "#BDBDBD",
}
RATES = ("0.0005", "0.001", "0.002")


def validated_data(path):
    payload = json.loads(path.read_text())
    jobs = {r["job"]["id"]: r for r in payload["jobs"]}
    cells = {}
    for cell in payload["cells"]:
        selected = [jobs[key] for key in cell["job_ids"]]
        scores = [r["test"] for job in selected for r in job["evaluations"]]
        assert len(scores) == cell["n_evaluations"]
        assert len(scores) == (10 if cell["method"] == "whole" else 30)
        assert all(
            sorted(r["seed"] for r in job["evaluations"]) == list(range(10)) for job in selected
        )
        assert math.isclose(statistics.mean(scores), cell["test_accuracy_mean"], abs_tol=1e-14)
        assert math.isclose(
            statistics.pstdev(scores), cell["test_accuracy_population_std"], abs_tol=1e-14
        )
        times = [job["timing"]["reduction_seconds"] for job in selected]
        assert times == cell["reduction_seconds"]
        assert statistics.median(times) == cell["median_reduction_seconds"]
        cells[cell["method"], cell["training_node_rate"]] = cell
    assert len(cells) == 13
    for method in METHODS[:-1]:
        assert all("MIG" not in jobs[key]["gpu_name"] for key in cells[method, "0.002"]["job_ids"])
    return payload, cells


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#C8C8C8",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "hatch.linewidth": 1.0,
            "svg.fonttype": "none",
            "svg.hashsalt": "graphslim-products",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="#E8E8E8", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.spines[["top", "right"]].set_visible(False)


def bar(ax, x, height, width, method, **kwargs):
    return ax.bar(
        x,
        height,
        width,
        color=COLORS[method],
        edgecolor="white",
        linewidth=0.8,
        hatch="//" if method == "clustgdd" else None,
        zorder=3,
        **kwargs,
    )


def save(fig, stem, output):
    paths = []
    for suffix in ("svg", "png", "pdf"):
        target = output / f"{stem}.{suffix}"
        metadata = (
            {"Date": None}
            if suffix == "svg"
            else {"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else {}
        )
        fig.savefig(target, dpi=200, metadata=metadata)
        paths.append(target)
    plt.close(fig)
    return paths


def results_figure(cells):
    fig, (accuracy, reduction) = plt.subplots(
        1, 2, figsize=(16, 6.0), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.23, top=0.74, wspace=0.22)
    fig.text(0.055, 0.95, "GraphSlim on ogbn-products", fontsize=23, weight="bold")
    legend = [
        Patch(
            facecolor=COLORS[m],
            edgecolor="white",
            hatch="//" if m == "clustgdd" else None,
            label=NAMES[m],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.89),
        ncol=5,
        frameon=True,
        edgecolor="#D3D3D3",
        fontsize=12.5,
        handlelength=2,
        columnspacing=2.0,
    )
    centers = np.arange(3) * 1.30
    width = 0.18
    for index, method in enumerate(METHODS):
        chosen = [cells[method, None if method == "whole" else rate] for rate in RATES]
        means = np.array([100 * c["test_accuracy_mean"] for c in chosen])
        errors = np.array([100 * c["test_accuracy_population_std"] for c in chosen])
        x = centers + (index - 2) * width
        bar(
            accuracy,
            x,
            means,
            width,
            method,
            yerr=errors,
            error_kw={"ecolor": "#555555", "elinewidth": 0.8, "capsize": 2},
        )
        for position, mean, error in zip(x, means, errors):
            accuracy.text(
                position,
                mean + error + 1.3,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="bold" if method == "clustgdd" else "normal",
            )
    accuracy.set_xticks(centers, ["0.05%", "0.1%", "0.2%"], weight="bold")
    accuracy.set_xlabel("Reduction budget (% of training nodes)", labelpad=12)
    accuracy.set_ylabel("Test accuracy (%)")
    accuracy.set_ylim(0, 85)
    accuracy.set_yticks(range(0, 81, 20))
    accuracy.set_title("GCN accuracy · higher is better", loc="left", pad=18)
    axes_style(accuracy)

    for index, method in enumerate(METHODS[:-1]):
        cell = cells[method, "0.002"]
        values = cell["reduction_seconds"]
        median = statistics.median(values)
        bar(
            reduction,
            index,
            median,
            0.65,
            method,
            yerr=[[median - min(values)], [max(values) - median]],
            error_kw={"ecolor": "#555555", "elinewidth": 0.9, "capsize": 3},
        )
        reduction.text(
            index,
            max(values) + 5,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold" if method == "clustgdd" else "normal",
        )
    reduction.set_xticks(range(4), [NAMES[m] for m in METHODS[:-1]], fontsize=10.5)
    reduction.set_ylabel("Graph reduction time (s)")
    reduction.set_ylim(0, 265)
    reduction.set_yticks(range(0, 251, 50))
    reduction.set_title("0.2% budget · lower is better", loc="left", pad=18)
    axes_style(reduction)
    fig.text(
        0.055,
        0.10,
        "Accuracy: mean ± population SD, 3 reduction seeds × 10 GCN seeds "
        "(full graph: 10 GCN seeds); official OGB split; 300 updates.",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.055,
        "Time: median with min–max across 3 reductions, including graph saving. "
        "GPU for timing: NVIDIA RTX PRO 6000 Blackwell Server Edition.",
        fontsize=10,
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "benchmark/results/products_20261028.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    options = parser.parse_args(argv)
    payload, cells = validated_data(options.input)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    paths = save(results_figure(cells), "products_results", options.output_dir)
    manifest = {
        "input_sha256": hashlib.sha256(options.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_reference": "https://github.com/deepseek-ai/DeepSeek-R1/blob/main/figures/benchmark.jpg",
        "accuracy_error_bars": "population standard deviation over 30 evaluations; full graph 10",
        "time_error_bars": "minimum to maximum of three reduction times; bar is the median",
        "timing_gpu": payload["hardware"]["gpu"],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    target = options.output_dir / "products_figures.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated 13 accuracy cells and 3 paired timing rows; wrote {len(paths)} figures.")


if __name__ == "__main__":
    main()

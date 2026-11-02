"""Draw measured ogbn-arxiv reduction time and peak GPU memory in blue/gray bars."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/results/arxiv_clustering_comparison_20261101.json"
REVIEWED = ROOT / "benchmark/results/arxiv_clustering_figure.json"
STEM = ROOT / "figures/arxiv_clustering_comparison"
METHODS = ("cgc", "clustgdd")
NAMES = {"cgc": "CGC", "clustgdd": "ClustGDD"}
RATES = ("0.001", "0.005", "0.01")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_data():
    source = json.loads(SOURCE.read_text())
    manifest = source["source_manifest"]
    assert manifest["data"]["nodes"] == 169343 and manifest["data"]["train"] == 90941
    assert manifest["seeds"] == [1, 2, 3] and manifest["rates"] == list(RATES)
    assert manifest["methods"] == list(METHODS)
    assert len(source["records"]) == 36 and len({r["id"] for r in source["records"]}) == 36
    cells = {(c["method"], c["rate"], c["backend"]): c for c in source["cells"]}
    pairs = []
    for rate in RATES:
        for method in METHODS:
            pair = dict(method=method, rate=rate, implementations={})
            for backend in ("author", "graphslim"):
                records = [
                    r
                    for r in source["records"]
                    if (r["method"], r["rate"], r["backend"]) == (method, rate, backend)
                ]
                assert {r["seed"] for r in records} == {1, 2, 3} and len(records) == 3
                assert all(
                    r["status"] == "complete" and len(r["evaluations"]) == 1 for r in records
                )
                assert all(r["input_sha256"] == manifest["input_sha256"] for r in records)
                assert all(r["environment"]["gpu_uuid"] == manifest["gpu"]["uuid"] for r in records)
                assert all(r["synthetic_nodes"] == int(90941 * float(rate)) for r in records)
                cell = cells[method, rate, backend]
                for field in ("reduction_seconds", "peak_cuda_allocated_bytes"):
                    values = [
                        r[field] if field == "reduction_seconds" else r["memory"][field]
                        for r in records
                    ]
                    assert all(math.isfinite(v) and v > 0 for v in values)
                    assert statistics.median(values) == cell[field]["median"]
                    assert (
                        min(values) == cell[field]["minimum"]
                        and max(values) == cell[field]["maximum"]
                    )
                accuracy = 100 * statistics.mean(
                    r["evaluations"][0]["test_accuracy"] for r in records
                )
                assert math.isclose(accuracy, cell["accuracy_mean"])
                pair["implementations"][backend] = cell
            a, b = (pair["implementations"][k] for k in ("author", "graphslim"))
            pair["time_ratio"] = a["reduction_seconds"]["median"] / b["reduction_seconds"]["median"]
            pair["memory_ratio"] = (
                a["peak_cuda_allocated_bytes"]["median"] / b["peak_cuda_allocated_bytes"]["median"]
            )
            pair["quality_shortfall_percent"] = (
                100 * max(0, a["accuracy_mean"] - b["accuracy_mean"]) / a["accuracy_mean"]
            )
            assert math.isclose(pair["time_ratio"], a["author_over_graphslim_time"])
            pairs.append(pair)
    archive = ROOT / source["source_archive"]["path"]
    assert sha(archive) == source["source_archive"]["sha256"]
    return dict(
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=sha(SOURCE),
        renderer_sha256=sha(__file__),
        source_archive=source["source_archive"],
        data=source["data"],
        protocol=source["protocol"],
        displayed_metrics=["reduction_seconds", "peak_cuda_allocated_bytes"],
        time_normalization="Author median divided by each implementation's times within each matched method/budget; author baseline 1x. Labels retain measured median seconds.",
        uncertainty="Bars are medians of three fresh-process reductions; whiskers show observed minimum to maximum.",
        style_reference="Established GraphSlim DeepSeek-inspired benchmark styling: white background, blue/gray grouped bars, hatched GraphSlim series, serif labels.",
        pairs=pairs,
    )


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    blue, gray, ink = "#4F6BFF", "#BDBDBD", "#202020"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#C8C8C8",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": ink,
            "axes.labelcolor": ink,
            "hatch.linewidth": 1.0,
            "svg.fonttype": "none",
            "svg.hashsalt": "graphslim-arxiv-efficiency",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.72, wspace=0.24)
    fig.text(0.065, 0.94, "GraphSlim on ogbn-arxiv", fontsize=27, weight="bold")
    fig.text(
        0.065,
        0.89,
        "169,343 nodes · full undirected graph · one RTX PRO 6000 Blackwell",
        fontsize=15,
        color="#525252",
    )
    fig.legend(
        handles=[
            Patch(facecolor=gray, edgecolor="white", label="Author implementation"),
            Patch(facecolor=blue, edgecolor="white", hatch="//", label="GraphSlim"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.865),
        ncol=2,
        frameon=True,
        edgecolor="#D3D3D3",
        fontsize=15,
        handlelength=2,
        columnspacing=2.8,
    )
    positions = np.array([0, 1, 2.4, 3.4, 4.8, 5.8])
    labels = [NAMES[p["method"]] + f"\n{100 * float(p['rate']):g}%" for p in data["pairs"]]
    for ax, field, scale, title, ylabel in [
        (
            axes[0],
            "reduction_seconds",
            1,
            "Reduction speed",
            "Speedup (×)",
        ),
        (
            axes[1],
            "peak_cuda_allocated_bytes",
            2**30,
            "Peak GPU memory",
            "GiB · PyTorch peak allocated",
        ),
    ]:

        def transformed(pair, backend):
            summary = pair["implementations"][backend][field]
            if field == "reduction_seconds":
                reference = pair["implementations"]["author"][field]["median"]
                return (
                    reference / summary["median"],
                    reference / summary["maximum"],
                    reference / summary["minimum"],
                )
            return tuple(summary[key] / scale for key in ("median", "minimum", "maximum"))

        maximum = max(transformed(p, b)[2] for p in data["pairs"] for b in ("author", "graphslim"))
        ax.set_title(title, pad=18)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_ylim(0, maximum * 1.24)
        ax.set_xticks(positions, labels, fontsize=12.5)
        ax.set_xlabel("Method / fraction of labeled training nodes", labelpad=15, fontsize=13)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", color="#E8E8E8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        for offset, backend, color, hatch in [
            (-0.18, "author", gray, None),
            (0.18, "graphslim", blue, "//"),
        ]:
            summaries = [p["implementations"][backend][field] for p in data["pairs"]]
            transformed_values = [transformed(p, backend) for p in data["pairs"]]
            medians = np.array([v[0] for v in transformed_values])
            lows = np.array([v[1] for v in transformed_values])
            highs = np.array([v[2] for v in transformed_values])
            ax.bar(
                positions + offset,
                medians,
                0.33,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                hatch=hatch,
                zorder=3,
                yerr=np.stack([medians - lows, highs - medians]),
                error_kw={"ecolor": "#555555", "elinewidth": 0.8, "capsize": 2},
            )
            for x, value, high, summary in zip(positions + offset, medians, highs, summaries):
                label = (
                    f"{summary['median']:.1f} s" if field == "reduction_seconds" else f"{value:.2f}"
                )
                if field == "reduction_seconds" and backend == "graphslim":
                    label += f"\n({value:.2f}×)"
                padding = (
                    0.10
                    if field == "peak_cuda_allocated_bytes" and backend == "graphslim"
                    else 0.022
                )
                ax.text(
                    x,
                    high + maximum * padding,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    weight="bold" if backend == "graphslim" else "normal",
                )
    fig.text(
        0.065,
        0.105,
        "3 reductions per bar; whiskers: observed min–max. Speed is relative to each author's median (1×); labels retain measured seconds.",
        fontsize=12.5,
        color="#525252",
    )
    fig.text(
        0.065,
        0.06,
        "0.1%, 0.5%, 1% of training nodes retain 90, 454, 909 nodes. Timing and GPU peaks exclude downstream GCN evaluation.",
        fontsize=12.5,
        color="#525252",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        for item in [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.texts,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]:
            bounds = item.get_window_extent(renderer)
            assert (
                bounds.x0 >= 0
                and bounds.x1 <= fig.bbox.width
                and bounds.y0 >= 0
                and bounds.y1 <= fig.bbox.height
            ), item.get_text()
    STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png", ".pdf"]:
        meta = (
            {"Date": None}
            if suffix == ".svg"
            else {"CreationDate": None, "ModDate": None}
            if suffix == ".pdf"
            else {}
        )
        fig.savefig(STEM.with_suffix(suffix), dpi=220, metadata=meta)
    plt.close(fig)
    svg = STEM.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = reviewed_data()
    if args.check:
        saved = json.loads(REVIEWED.read_text())
        hashes = saved.pop("assets_sha256")
        assert saved == data, "Source or renderer changed; regenerate the figure"
        for ext, expected in hashes.items():
            assert sha(STEM.with_suffix(ext)) == expected
        print("All 36 ogbn-arxiv records, six pairs and exported figures match.")
    else:
        plot(data)
        data["assets_sha256"] = {
            ext: sha(STEM.with_suffix(ext)) for ext in [".svg", ".png", ".pdf"]
        }
        REVIEWED.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print("Wrote DeepSeek-inspired ogbn-arxiv efficiency figures and reviewed data.")


if __name__ == "__main__":
    main()

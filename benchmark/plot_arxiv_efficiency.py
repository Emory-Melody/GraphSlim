"""Draw measured ogbn-arxiv reduction time and peak GPU memory in blue/open bars."""

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
            pair["relative_time"] = (
                b["reduction_seconds"]["median"] / a["reduction_seconds"]["median"]
            )
            pair["memory_ratio"] = (
                a["peak_cuda_allocated_bytes"]["median"] / b["peak_cuda_allocated_bytes"]["median"]
            )
            pair["quality_relative_gap_percent"] = (
                100 * abs(a["accuracy_mean"] - b["accuracy_mean"]) / abs(a["accuracy_mean"])
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
        metric_directions={"reduction_seconds": "lower", "peak_cuda_allocated_bytes": "lower"},
        time_normalization="Each implementation's times divided by the author median within each matched method/budget; author baseline 1. Lower is better; labels retain measured median seconds.",
        uncertainty="Bars are medians of three fresh-process reductions; whiskers show observed minimum to maximum.",
        style_reference="Established GraphSlim DeepSeek-inspired benchmark styling: white background, blue/open grouped bars, hatched GraphSlim series, serif labels.",
        pairs=pairs,
    )


def plot(data, paper=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import Bbox
    import numpy as np

    blue, gray, ink = "#4F6BFF", "#FFFFFF", "#000000"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.9,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
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
    if paper:
        fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.6))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.73, hspace=0.78)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18.5, 7.5))
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.72, wspace=0.24)
    fig.text(0.065, 0.94, "GraphSlim on ogbn-arxiv", fontsize=24 if paper else 27, weight="bold")
    fig.text(0.985, 0.94, "Lower is better ↓", fontsize=16, ha="right", color="#000000")
    fig.text(
        0.065,
        0.89,
        "169,343 nodes · full undirected graph · one RTX PRO 6000 Blackwell",
        fontsize=12 if paper else 15,
        color="#000000",
    )
    fig.legend(
        handles=[
            Patch(facecolor=gray, edgecolor="black", label="Author implementation"),
            Patch(facecolor=blue, edgecolor="black", hatch="//", label="GraphSlim"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.865),
        ncol=2,
        frameon=False,
        edgecolor="#000000",
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
            "Reduction time ↓",
            "Relative time · author = 1",
        ),
        (
            axes[1],
            "peak_cuda_allocated_bytes",
            2**30,
            "Peak GPU memory ↓",
            "GiB · PyTorch peak allocated",
        ),
    ]:

        def transformed(pair, backend):
            summary = pair["implementations"][backend][field]
            if field == "reduction_seconds":
                reference = pair["implementations"]["author"][field]["median"]
                return (
                    summary["median"] / reference,
                    summary["minimum"] / reference,
                    summary["maximum"] / reference,
                )
            return tuple(summary[key] / scale for key in ("median", "minimum", "maximum"))

        maximum = max(transformed(p, b)[2] for p in data["pairs"] for b in ("author", "graphslim"))
        ax.set_title(title, pad=18)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_ylim(0, maximum * 1.24)
        ax.set_xticks(positions, labels, fontsize=12.5)
        ax.set_xlabel("Method / fraction of labeled training nodes", labelpad=15, fontsize=13)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", color="#DCE5FF", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        # Leave room for the two-line labels beside the neighboring open bars.
        for offset, backend, color, hatch in [
            (-0.25, "author", gray, None),
            (0.25, "graphslim", blue, "//"),
        ]:
            summaries = [p["implementations"][backend][field] for p in data["pairs"]]
            transformed_values = [transformed(p, backend) for p in data["pairs"]]
            medians = np.array([v[0] for v in transformed_values])
            lows = np.array([v[1] for v in transformed_values])
            highs = np.array([v[2] for v in transformed_values])
            ax.bar(
                positions + offset,
                medians,
                0.24,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                hatch=hatch,
                zorder=3,
                yerr=np.stack([medians - lows, highs - medians]),
                error_kw={"ecolor": "#000000", "elinewidth": 0.8, "capsize": 2},
            )
            for x, value, high, summary in zip(positions + offset, medians, highs, summaries):
                label = (
                    f"{summary['median']:.1f} s" if field == "reduction_seconds" else f"{value:.2f}"
                )
                if field == "reduction_seconds" and backend == "graphslim":
                    label += f"\n({value:.2f}×)"
                ax.annotate(
                    label,
                    xy=(x, high),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11 if paper else 12,
                    weight="bold" if backend == "graphslim" else "normal",
                )
    if paper:
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Time / author median", fontsize=13)
        axes[1].set_ylabel("Peak GPU memory (GiB)", fontsize=13)
    fig.text(
        0.065,
        0.105,
        (
            "3 reductions per bar; whiskers: observed min–max.\nTime is relative to the author's median (1); both metrics: lower is better."
            if paper
            else "3 reductions per bar; whiskers: observed min–max. Time is relative to each author's median (1); lower is better for both metrics."
        ),
        fontsize=10 if paper else 12.5,
        color="#000000",
    )
    fig.text(
        0.065,
        0.06,
        (
            "0.1%, 0.5%, 1% of training nodes retain 90, 454, 909 nodes.\nTiming and GPU peaks exclude downstream GCN evaluation."
            if paper
            else "0.1%, 0.5%, 1% of training nodes retain 90, 454, 909 nodes. Timing and GPU peaks exclude downstream GCN evaluation."
        ),
        fontsize=10 if paper else 12.5,
        color="#000000",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if paper:
        assert (
            not axes[0]
            .yaxis.label.get_window_extent(renderer)
            .overlaps(axes[1].yaxis.label.get_window_extent(renderer))
        )
        assert (
            not axes[1]
            .xaxis.label.get_window_extent(renderer)
            .overlaps(fig.texts[-2].get_window_extent(renderer))
        )
    for ax in axes:
        label_bounds = [item.get_window_extent(renderer) for item in ax.texts]
        whisker_bounds = []
        for container in ax.containers:
            if not hasattr(container, "errorbar"):
                continue
            _, caps, collections = container.errorbar.lines
            for cap in caps:
                width = cap.get_markersize() * fig.dpi / 72
                height = cap.get_markeredgewidth() * fig.dpi / 72
                whisker_bounds.extend(
                    Bbox.from_bounds(x - width / 2, y - height / 2, width, height)
                    for x, y in cap.get_transform().transform(cap.get_xydata())
                )
            whisker_bounds.extend(
                Bbox(collection.get_transform().transform(segment)).padded(fig.dpi / 144)
                for collection in collections
                for segment in collection.get_segments()
            )
        for index, bounds in enumerate(label_bounds):
            assert not any(
                bounds.overlaps(bar.get_window_extent(renderer)) for bar in ax.patches
            ), f"Label overlaps a bar: {ax.texts[index].get_text()}"
            assert not any(bounds.overlaps(whisker) for whisker in whisker_bounds), (
                f"Label overlaps an error bar: {ax.texts[index].get_text()}"
            )
            assert not any(bounds.overlaps(other) for other in label_bounds[index + 1 :]), (
                f"Labels overlap: {ax.texts[index].get_text()}"
            )
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
    for suffix in ["_paper.pdf"] if paper else [".svg", ".png", ".pdf"]:
        meta = (
            {"Date": None}
            if suffix == ".svg"
            else {"CreationDate": None, "ModDate": None}
            if suffix.endswith(".pdf")
            else {}
        )
        fig.savefig(Path(str(STEM) + suffix), dpi=220, metadata=meta)
    plt.close(fig)
    svg = STEM.with_suffix(".svg")
    if paper:
        return
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
            assert sha(Path(str(STEM) + ext)) == expected
        print("All 36 ogbn-arxiv records, six pairs and exported figures match.")
    else:
        plot(data)
        plot(data, paper=True)
        data["assets_sha256"] = {
            ext: sha(Path(str(STEM) + ext)) for ext in [".svg", ".png", ".pdf", "_paper.pdf"]
        }
        REVIEWED.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print("Wrote DeepSeek-inspired ogbn-arxiv efficiency figures and reviewed data.")


if __name__ == "__main__":
    main()

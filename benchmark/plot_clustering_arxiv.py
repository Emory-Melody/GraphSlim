# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Render matched complete-reducer GECC/CGC/ClustGDD checks on ogbn-arxiv."""

import argparse
import json
import math
from pathlib import Path
import statistics

from benchmark.plot_arxiv_efficiency import ROOT, RATES, reviewed_data as earlier_data, sha

SOURCE = ROOT / "benchmark/results/gecc_arxiv_comparison_20261102.json"
REVIEWED = ROOT / "benchmark/results/clustering_arxiv_figure.json"
STEM = ROOT / "figures/clustering_arxiv_comparison"
METHODS = ("gecc", "cgc", "clustgdd")
NAMES = {"gecc": "GECC", "cgc": "CGC", "clustgdd": "ClustGDD"}


def reviewed_data():
    previous = earlier_data()
    source = json.loads(SOURCE.read_text())
    manifest = source["source_manifest"]
    assert manifest["methods"] == ["gecc"]
    assert manifest["seeds"] == [1, 2, 3] and manifest["rates"] == list(RATES)
    assert len(source["records"]) == 18
    assert len({r["id"] for r in source["records"]}) == 18
    assert len(source["paired_output_parity"]) == 9
    assert all(
        p["exact"] and p["max_abs"] == [0.0, 0.0, 0.0] for p in source["paired_output_parity"]
    )
    old_source = json.loads((ROOT / previous["source"]).read_text())
    old_manifest = old_source["source_manifest"]
    assert manifest["input_sha256"] == old_manifest["input_sha256"]
    assert manifest["gpu"] == old_manifest["gpu"]
    assert manifest["data"] == old_manifest["data"]
    assert sha(ROOT / source["source_archive"]["path"]) == source["source_archive"]["sha256"]
    pairs = {(p["method"], p["rate"]): p for p in previous["pairs"]}
    for rate in RATES:
        pair = dict(method="gecc", rate=rate, implementations={})
        for backend in ("author", "graphslim"):
            records = [r for r in source["records"] if (r["rate"], r["backend"]) == (rate, backend)]
            assert len(records) == 3 and {r["seed"] for r in records} == {1, 2, 3}
            assert all(r["status"] == "complete" and len(r["evaluations"]) == 1 for r in records)
            assert all(r["input_sha256"] == manifest["input_sha256"] for r in records)
            assert all(r["environment"]["gpu_uuid"] == manifest["gpu"]["uuid"] for r in records)
            assert all(r["synthetic_nodes"] == int(90941 * float(rate)) for r in records)
            cell = next(c for c in source["cells"] if (c["rate"], c["backend"]) == (rate, backend))
            for field in ("reduction_seconds", "peak_cuda_allocated_bytes"):
                values = [
                    r[field] if field == "reduction_seconds" else r["memory"][field]
                    for r in records
                ]
                assert all(math.isfinite(v) and v > 0 for v in values)
                assert cell[field]["median"] == statistics.median(values)
                assert (cell[field]["minimum"], cell[field]["maximum"]) == (
                    min(values),
                    max(values),
                )
            accuracy = 100 * statistics.mean(r["evaluations"][0]["test_accuracy"] for r in records)
            assert math.isclose(accuracy, cell["accuracy_mean"])
            pair["implementations"][backend] = cell
        a, b = [pair["implementations"][backend] for backend in ("author", "graphslim")]
        pair["relative_time"] = b["reduction_seconds"]["median"] / a["reduction_seconds"]["median"]
        pair["quality_relative_gap_percent"] = (
            100 * abs(a["accuracy_mean"] - b["accuracy_mean"]) / abs(a["accuracy_mean"])
        )
        pairs["gecc", rate] = pair
    return dict(
        renderer_sha256=sha(__file__),
        sources=[
            dict(
                path=previous["source"],
                sha256=previous["source_sha256"],
                archive=previous["source_archive"],
            ),
            dict(
                path=str(SOURCE.relative_to(ROOT)),
                sha256=sha(SOURCE),
                archive=source["source_archive"],
            ),
        ],
        data=source["data"],
        gpu=manifest["gpu"],
        input_sha256=manifest["input_sha256"],
        record_count=54,
        pair_count=9,
        methods=list(METHODS),
        rates=list(RATES),
        displayed_metrics=["reduction_seconds", "peak_cuda_allocated_bytes"],
        metric_directions={"reduction_seconds": "lower", "peak_cuda_allocated_bytes": "lower"},
        scope="Matched complete reductions of the same full ogbn-arxiv tensors on the same GPU; two frozen measurement cohorts, with GECC added separately. Archived synthetic GECC operator checks are outside this figure.",
        uncertainty="Bars are medians of three fresh-process reductions; whiskers show observed minimum to maximum.",
        gecc_initialization=manifest["protocol"]["initialization"],
        gecc_paired_output_parity=source["paired_output_parity"],
        pairs=[pairs[method, rate] for method in METHODS for rate in RATES],
    )


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import Bbox
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "hatch.linewidth": 1.0,
            "svg.fonttype": "none",
            "svg.hashsalt": "graphslim-clustering-arxiv",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.76, hspace=0.66, wspace=0.33)
    fig.text(0.055, 0.95, "GraphSlim on ogbn-arxiv", fontsize=25, weight="bold")
    fig.text(0.985, 0.95, "Lower is better ↓", fontsize=14, ha="right")
    fig.text(
        0.055,
        0.90,
        "169,343 nodes · full undirected graph · same RTX PRO 6000 Blackwell",
        fontsize=12,
    )
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="Author implementation"),
            Patch(facecolor="#4F6BFF", edgecolor="black", hatch="//", label="GraphSlim"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.865),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2,
        columnspacing=3,
    )
    positions = np.arange(3)
    for column, method in enumerate(METHODS):
        pairs = [p for p in data["pairs"] if p["method"] == method]
        for row, field in enumerate(("reduction_seconds", "peak_cuda_allocated_bytes")):
            ax = axes[row, column]

            def transformed(pair, backend):
                values = pair["implementations"][backend][field]
                scale = pair["implementations"]["author"][field]["median"] if row == 0 else 2**30
                return tuple(values[key] / scale for key in ("median", "minimum", "maximum"))

            maximum = max(transformed(p, b)[2] for p in pairs for b in ("author", "graphslim"))
            for backend, offset, color, hatch in [
                ("author", -0.25, "white", None),
                ("graphslim", 0.25, "#4F6BFF", "//"),
            ]:
                values = np.array([transformed(p, backend) for p in pairs])
                medians, lows, highs = values.T
                ax.bar(
                    positions + offset,
                    medians,
                    0.24,
                    color=color,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=0.8,
                    zorder=3,
                    yerr=np.stack([medians - lows, highs - medians]),
                    error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
                )
                for x, value, high, pair in zip(positions + offset, medians, highs, pairs):
                    label = (
                        f"{pair['implementations'][backend][field]['median']:.1f} s"
                        if row == 0
                        else f"{value:.2f}"
                    )
                    if row == 0 and backend == "graphslim":
                        label += f"\n({value:.2f}×)"
                    # GECC times are close: put its two-line labels above the
                    # author labels rather than squeezing either font size.
                    padding = 5
                    if row == 0 and method == "gecc" and backend == "graphslim":
                        high = maximum
                        padding = 23
                    ax.annotate(
                        label,
                        xy=(x, high),
                        xytext=(0, padding),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        weight="bold" if backend == "graphslim" else "normal",
                    )
            ax.set_title(f"({chr(97 + row * 3 + column)}) {NAMES[method]}", pad=14)
            ax.set_ylim(0, maximum * (1.85 if row == 0 and method == "gecc" else 1.52))
            ax.set_xticks(positions, ["0.1%", "0.5%", "1%"], fontsize=12)
            ax.set_ylabel("Time / author median" if row == 0 else "Peak GPU memory (GiB)")
            if row == 1:
                ax.set_xlabel("Training-node budget", labelpad=10)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(axis="both", length=0, pad=6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", color="#DCE5FF", linewidth=0.8)
    fig.text(
        0.055,
        0.10,
        "3 fresh-process reductions per bar; whiskers: observed min–max. Timing and GPU peaks exclude GCN evaluation.",
        fontsize=10.5,
    )
    fig.text(
        0.055,
        0.055,
        "Budgets retain 90 / 454 / 909 nodes. GECC uses matched author initialization; reduced features, adjacency and labels match exactly.",
        fontsize=10.5,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        labels = [item.get_window_extent(renderer) for item in ax.texts]
        whiskers = []
        for container in ax.containers:
            if not hasattr(container, "errorbar"):
                continue
            _, caps, collections = container.errorbar.lines
            for cap in caps:
                width = cap.get_markersize() * fig.dpi / 72
                height = cap.get_markeredgewidth() * fig.dpi / 72
                whiskers.extend(
                    Bbox.from_bounds(x - width / 2, y - height / 2, width, height)
                    for x, y in cap.get_transform().transform(cap.get_xydata())
                )
            whiskers.extend(
                Bbox(collection.get_transform().transform(segment)).padded(fig.dpi / 144)
                for collection in collections
                for segment in collection.get_segments()
            )
        for i, bounds in enumerate(labels):
            assert not any(
                bounds.overlaps(bar.get_window_extent(renderer)) for bar in ax.patches
            ), ax.texts[i].get_text()
            assert not any(bounds.overlaps(other) for other in labels[i + 1 :]), ax.texts[
                i
            ].get_text()
            assert not any(bounds.overlaps(whisker) for whisker in whiskers), ax.texts[i].get_text()
        for item in [
            *ax.texts,
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
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
    for item in fig.texts:
        bounds = item.get_window_extent(renderer)
        assert (
            bounds.x0 >= 0
            and bounds.x1 <= fig.bbox.width
            and bounds.y0 >= 0
            and bounds.y1 <= fig.bbox.height
        ), item.get_text()
    for ext in (".svg", ".png", ".pdf", "_paper.pdf"):
        metadata = (
            {"Date": None}
            if ext == ".svg"
            else {"CreationDate": None, "ModDate": None}
            if ext.endswith(".pdf")
            else {}
        )
        fig.savefig(Path(str(STEM) + ext), dpi=220, metadata=metadata)
    plt.close(fig)
    svg = STEM.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    data = reviewed_data()
    if options.check:
        saved = json.loads(REVIEWED.read_text())
        assets = saved.pop("assets_sha256")
        assert saved == data
        assert all(sha(Path(str(STEM) + ext)) == digest for ext, digest in assets.items())
        print(
            "54 complete reductions, nine matched pairs and all combined figure exports verified."
        )
    else:
        plot(data)
        data["assets_sha256"] = {
            ext: sha(Path(str(STEM) + ext)) for ext in (".svg", ".png", ".pdf", "_paper.pdf")
        }
        REVIEWED.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print("Wrote combined GECC/CGC/ClustGDD ogbn-arxiv figure.")


if __name__ == "__main__":
    main()

"""Plot cumulative reproduction coverage for documented numerical references.

The 110 method--dataset pairs exclude Clustering, CentP, GDEM and other missing references.
Measured shortfall bands are disjoint. Both the README and paper count scores
at or above the reference as zero shortfall; two-sided errors remain optional.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/results/reproduction_coverage_20261106.json"
REVIEWED = ROOT / "benchmark/results/reproduction_coverage_figure.json"
STEM = ROOT / "figures/reproduction_coverage"
METRIC = "mean_shortfall_percent"
THRESHOLDS = (1, 2, 5, 10)
COLORS = ("#4F6BFF", "#7E95FF", "#AAB9FF", "#D4DDFF", "#FFFFFF")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_data(metric=METRIC):
    source = json.loads(SOURCE.read_text())
    pairs = source["pairs"]
    denominator = source["total_pairs"]
    assert len(pairs) == len({(p["method"], p["dataset"]) for p in pairs}) == denominator
    assert len({p["method"] for p in pairs}) == source["method_count"]
    assert {p["dataset"] for p in pairs} == set(source["datasets"])
    assert sum(p["complete_result"] for p in pairs) == source["complete_pairs"]
    assert not any(p["credited_by_policy"] for p in pairs)
    assert all(
        p["method"] != "clustering" and p["reference_status"] == "numeric_reference" for p in pairs
    )
    assert len(pairs) == source["numeric_reference_pairs"]
    assert all(math.isfinite(p[metric]) and p[metric] >= 0 for p in pairs)
    measure = "shortfall" if metric == "mean_shortfall_percent" else "relative error"

    def keys(selected):
        return sorted(f"{p['method']}/{p['dataset']}" for p in selected)

    segments = []
    lower = 0
    for upper, color in zip(THRESHOLDS, COLORS):
        members = [p for p in pairs if p["complete_result"] and lower <= p[metric] < upper]
        segments.append(
            dict(
                kind=metric,
                lower_inclusive=lower,
                upper_exclusive=upper,
                label=f"Measured {measure} <1%"
                if lower == 0
                else f"Measured {measure} {lower}–<{upper}%",
                count=len(members),
                pairs=keys(members),
                color=color,
            )
        )
        lower = upper
    columns = []
    for threshold in THRESHOLDS:
        counts = [s["count"] if s["upper_exclusive"] <= threshold else 0 for s in segments]
        measured_count = sum(p["complete_result"] and p[metric] < threshold for p in pairs)
        total = sum(counts)
        expected = source["thresholds"][metric].get(str(threshold))
        assert measured_count == total
        if expected is not None:
            assert total == expected["count"] and expected["denominator"] == denominator
            assert math.isclose(100 * total / denominator, expected["percent"])
        columns.append(
            dict(
                threshold_percent=threshold,
                segment_counts=counts,
                measured_count=measured_count,
                count=total,
                denominator=denominator,
                percent=100 * total / denominator,
            )
        )
    outside = [p for p in pairs if not p["complete_result"] or p[metric] >= THRESHOLDS[-1]]
    assert sum(s["count"] for s in segments) + len(outside) == denominator
    return dict(
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=sha(SOURCE),
        renderer_sha256=sha(__file__),
        as_of_utc=source["as_of_utc"],
        scope=source["scope"],
        unit=source["unit"],
        metric=metric,
        formula=source["formulas"][metric],
        threshold_rule=source["threshold_rule"],
        total_pairs=denominator,
        numeric_reference_pairs=len(pairs),
        method_count=source["method_count"],
        dataset_count=len(source["datasets"]),
        shared_observation_pairs=source.get("cohort_expansion", {}).get(
            "shared_observation_pairs", 0
        ),
        segments=segments,
        columns=columns,
        outside_largest_threshold=keys(outside),
        interpretation="Each column is cumulative for the same roster of documented numerical comparisons. Measured ranges are disjoint; missing references and completion credits are excluded. Eight added GC4NC pairs reuse the archived observations reported by the revised reference tables; they are not independent replication runs.",
    )


def plot(data, stem=STEM):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    ink = "#000000"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.9,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "hatch.linewidth": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "graphslim-reproduction-coverage",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.82)
    handles = [
        Patch(
            facecolor=s["color"],
            edgecolor="white",
            label=s["label"],
        )
        for s in data["segments"]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=2,
        frameon=False,
        edgecolor="#000000",
        fontsize=12,
        handlelength=2.1,
        columnspacing=2.3,
    )
    positions = np.arange(len(data["columns"])) * 1.25
    base = np.zeros(len(data["columns"]))
    denominator = data["total_pairs"]
    for index, segment in enumerate(data["segments"]):
        counts = np.array([c["segment_counts"][index] for c in data["columns"]])
        heights = 100 * counts / denominator
        ax.bar(
            positions,
            heights,
            0.68,
            bottom=base,
            color=segment["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for x, height, bottom, count in zip(positions, heights, base, counts):
            if count and height < 3:
                ax.annotate(
                    str(count),
                    xy=(x + 0.34, bottom + height / 2),
                    xytext=(x + 0.46, bottom + height / 2),
                    ha="left",
                    va="center",
                    fontsize=11,
                    weight="bold",
                    arrowprops=dict(arrowstyle="-", color=ink, linewidth=0.8),
                )
            elif count:
                ax.text(
                    x,
                    bottom + height / 2,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if index == 0 else ink,
                    fontsize=9 if height < 6 else 16,
                    weight="bold",
                )
        base += heights
    for x, column in zip(positions, data["columns"]):
        ax.text(
            x,
            column["percent"] + 2.2,
            f"{column['percent']:.2f}%\n{column['count']}/{denominator}",
            ha="center",
            va="bottom",
            fontsize=16,
            weight="bold",
        )
    ax.set_xticks(
        positions,
        [f"<{c['threshold_percent']}%" for c in data["columns"]],
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel(
        "Allowed mean relative shortfall"
        if data["metric"] == "mean_shortfall_percent"
        else "Allowed mean two-sided relative error",
        labelpad=14,
    )
    ax.set_ylabel("Coverage of fixed roster (%)", labelpad=10)
    ax.set_ylim(0, 116)
    ax.set_xlim(-0.75, positions[-1] + 0.75)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="#DCE5FF", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in [
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.texts,
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
        *fig.texts,
    ]:
        bounds = item.get_window_extent(renderer)
        assert (
            bounds.x0 >= 0
            and bounds.x1 <= fig.bbox.width
            and bounds.y0 >= 0
            and bounds.y1 <= fig.bbox.height
        ), item.get_text()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png", ".pdf"]:
        meta = (
            {"Date": None}
            if suffix == ".svg"
            else {"CreationDate": None, "ModDate": None}
            if suffix == ".pdf"
            else {}
        )
        fig.savefig(stem.with_suffix(suffix), dpi=220, metadata=meta)
    plt.close(fig)
    path = stem.with_suffix(".svg")
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    data = reviewed_data()
    if options.check:
        saved = json.loads(REVIEWED.read_text())
        hashes = saved.pop("assets_sha256")
        assert saved == data, "Source or renderer changed; regenerate the coverage chart"
        for suffix, expected in hashes.items():
            assert sha(STEM.with_suffix(suffix)) == expected
        print(
            f"Cumulative coverage chart matches all {data['total_pairs']} documented-reference pairs and disjoint measured bands."
        )
    else:
        plot(data)
        data["assets_sha256"] = {
            ext: sha(STEM.with_suffix(ext)) for ext in [".svg", ".png", ".pdf"]
        }
        REVIEWED.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print("Wrote cumulative coverage chart, SVG/PNG/PDF and reviewed counts.")


if __name__ == "__main__":
    main()

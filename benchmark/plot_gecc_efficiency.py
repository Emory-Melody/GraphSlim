"""Render archived GECC operator checks, separately from full-reducer comparisons."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "benchmark/results/gecc_operator_figure.json"
STEM = ROOT / "figures/gecc_operator_comparison"
OPERATORS = ("GECC bounded squared distances", "GECC streamed weighted hops")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_data():
    pairs, sources = [], []
    for device in ("cpu", "gpu"):
        path = ROOT / f"benchmark/results/gecc_operator_{device}_20261029.json"
        report = json.loads(path.read_text())
        assert report["device"] == ("cpu" if device == "cpu" else "cuda:0")
        parity = next(r for r in report["reducers"]["checks"] if r["method"] == "gecc")
        assert parity["passed"] and parity["max_abs"] == [0.0, 0.0, 0.0]
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            measured_utc=report["utc"], source_sha256=report["source_sha256"]))
        for name in OPERATORS:
            row = next(r for r in report["operators"] if r["operator"] == name)
            assert row["passed"]
            for backend in ("reference", "optimized"):
                values = row[backend]["seconds"]
                assert len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values)
                assert statistics.median(values) == row[backend]["median_seconds"]
            ratio = row["optimized"]["median_seconds"] / row["reference"]["median_seconds"]
            assert math.isclose(1 / ratio, row["speedup"])
            pairs.append(dict(device=device, operator=name, reference=row["reference"],
                              optimized=row["optimized"], relative_time=ratio,
                              speedup=row["speedup"], numerical_check_passed=True))
    return dict(
        sources=sources, renderer_sha256=sha(__file__), pairs=pairs,
        scope="Archived isolated operators on synthetic fixed tensors, measured 2026-10-29; no full-GECC or matched-author implementation speedup is established.",
        inputs={"distances": "4096 points x 64 features, 128 centers; hard-clustering distance kernel",
                "weighted_hops": "2048 nodes, 64 features, four sparse propagation hops"},
        protocol="Four CPU threads; one warmup followed by three timed calls in the same process; CUDA synchronized before and after timing. GPU model and UUID were not recorded in these reports.",
        memory="CUDA memory is maximum additional allocated bytes above resident inputs across timed calls. Per-repeat memory values and CPU memory were not retained.",
        uncertainty="Time whiskers show observed min-max, not confidence intervals. Memory has no interval.",
        numerical_validation="The two operator outputs matched their reference exactly; archived small-graph GECC reducer checks also reported zero tensor difference.",
    )


def plot(data, paper=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 13,
                         "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black",
                         "svg.fonttype": "none", "svg.hashsalt": "graphslim-gecc-operators",
                         "pdf.fonttype": 42, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    if paper:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8.6))
        fig.subplots_adjust(left=.12, right=.97, bottom=.24, top=.72, hspace=.75)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
        fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.70, wspace=.30)
    fig.text(.075, .94, "GECC operator checks", weight="bold", fontsize=24)
    fig.text(.075, .88, "Archived operator measurements · 29 October 2026", fontsize=13)
    fig.legend(handles=[Patch(facecolor="white", edgecolor="black", label="Reference operator"),
                        Patch(facecolor="#4F6BFF", edgecolor="black", hatch="//", label="GraphSlim operator")],
               loc="upper center", bbox_to_anchor=(.55, .84), ncol=2, frameon=False)
    for ax, field in zip(axes, ("time", "memory")):
        pairs = data["pairs"] if field == "time" else [p for p in data["pairs"] if p["device"] == "gpu"]
        positions = np.arange(len(pairs))
        largest = 1.0
        for backend, offset, color, hatch in [("reference", -.18, "white", None),
                                               ("optimized", .18, "#4F6BFF", "//")]:
            heights, lows, highs, labels = [], [], [], []
            for pair in pairs:
                row, ref = pair[backend], pair["reference"]
                if field == "time":
                    denom = ref["median_seconds"]
                    value = row["median_seconds"] / denom
                    lo, hi = min(row["seconds"]) / denom, max(row["seconds"]) / denom
                    labels.append(f"{1000000 * row['median_seconds']:.1f} µs" if row["median_seconds"] < .001 else f"{1000 * row['median_seconds']:.2f} ms")
                else:
                    value = row["peak_additional_gpu_bytes"] / ref["peak_additional_gpu_bytes"]
                    lo = hi = value
                    labels.append(f"{row['peak_additional_gpu_bytes'] / 2**20:.2f} MiB")
                heights.append(value); lows.append(value - lo); highs.append(hi - value)
                largest = max(largest, hi)
            bars = ax.bar(positions + offset, heights, .34, color=color, edgecolor="black",
                          hatch=hatch, linewidth=.8, zorder=3,
                          yerr=[lows, highs] if field == "time" else None,
                          error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 1})
            for bar, label, hi in zip(bars, labels, highs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + hi + .04,
                        label, ha="center", va="bottom", fontsize=11 if paper else 9, rotation=0)
        ax.set_ylim(0, largest * 1.45)
        ax.set_title("Operator time ↓" if field == "time" else "Additional GPU memory ↓", weight="bold", pad=14)
        ax.set_ylabel("Relative time · reference = 1" if field == "time" else "Relative peak · reference = 1")
        ax.set_xticks(positions, [("Distances" if "distances" in p["operator"] else "Weighted hops") + "\n" + p["device"].upper() for p in pairs], fontsize=12 if paper else 10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", color="#DCE5FF", zorder=0)
        ax.set_axisbelow(True)
    if paper:
        axes[0].set_ylabel("Time / reference", fontsize=13)
        axes[1].set_ylabel("GPU peak / reference", fontsize=13)
    fig.text(.075, .11, ("Time: median of 3 calls after warmup; whiskers: observed min–max.\nMemory: additional CUDA allocation."
                 if paper else "Time: median of 3 calls after warmup; whiskers: observed min–max. Memory: additional CUDA allocation."), fontsize=10)
    fig.text(.075, .055, "Scope: isolated kernels; complete GECC runtime and accuracy are not compared here.", fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if paper:
        assert not axes[0].yaxis.label.get_window_extent(renderer).overlaps(axes[1].yaxis.label.get_window_extent(renderer))
    for item in [*fig.texts, *(t for ax in axes for t in [*ax.texts, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels()])]:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width and box.y0 >= 0 and box.y1 <= fig.bbox.height, item.get_text()
    STEM.parent.mkdir(parents=True, exist_ok=True)
    for ext in (["_paper.pdf"] if paper else [".svg", ".png", ".pdf"]):
        metadata = {"Date": None} if ext == ".svg" else {"CreationDate": None, "ModDate": None} if ext.endswith(".pdf") else {}
        fig.savefig(Path(str(STEM) + ext), dpi=220, metadata=metadata)
    plt.close(fig)
    if not paper:
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
        assert saved == data
        assert all(sha(Path(str(STEM) + ext)) == digest for ext, digest in hashes.items())
        print("GECC figure matches all four archived operator pairs and their exact numerical checks.")
    else:
        plot(data)
        plot(data, paper=True)
        data["assets_sha256"] = {ext: sha(Path(str(STEM) + ext)) for ext in (".svg", ".png", ".pdf", "_paper.pdf")}
        REVIEWED.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

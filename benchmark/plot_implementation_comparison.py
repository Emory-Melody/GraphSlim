"""Render the README's measured time, peak-memory, and accuracy comparisons.

Uses the complete frozen six-pair roster. Run with ``--check`` to validate the
reviewed chart data and exported figures without executing a benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/results/implementation_comparison_20261031.json"
DATA = ROOT / "benchmark/results/implementation_comparison_figure.json"
STEM = ROOT / "figures/implementation_comparison"
ROSTER = [(m, d) for m in ("cgc", "clustgdd", "gcgp") for d in ("cora", "citeseer")]
NAMES = {"cgc": "CGC", "clustgdd": "ClustGDD", "gcgp": "GCGP"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spread(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def quality(author, package):
    return {
        "author_percent": author,
        "graphslim_percent": package,
        "relative_shortfall_percent": 100 * max(0.0, author - package) / author,
        "absolute_relative_gap_percent": 100 * abs(author - package) / author,
    }


def reviewed_data():
    source = json.loads(SOURCE.read_text())
    rows = {(r["method"], r["dataset"], r["backend"]): r for r in source["rows"]}
    assert len(rows) == 12 and len(source["jobs"]) == 36
    pairs = []
    for method, dataset in ROSTER:
        pair = dict(method=method, dataset=dataset, implementations={})
        scores = {}
        for backend in ("author", "graphslim"):
            row = rows[method, dataset, backend]
            jobs = [
                j
                for j in source["jobs"]
                if (j["method"], j["dataset"], j["backend"]) == (method, dataset, backend)
            ]
            assert len(jobs) == 3 and {j["seed"] for j in jobs} == {1, 2, 3}
            assert all(j["status"] == "complete" for j in jobs)
            tests = [100 * e["test_accuracy"] for j in jobs for e in j["evaluations"]]
            assert len(tests) == 30
            assert math.isclose(statistics.mean(tests), row["accuracy_mean"])
            assert math.isclose(statistics.pstdev(tests), row["accuracy_std"])
            scores[backend] = statistics.mean(tests)
            metrics = {
                "time_seconds": spread([j["reduction_seconds"] for j in jobs]),
                "peak_process_rss_mib": spread(
                    [j["memory"]["peak_sampled_rss_bytes"] / 1024**2 for j in jobs]
                ),
                "peak_cuda_allocated_mib": None
                if row["device"] == "cpu"
                else spread([j["memory"]["peak_cuda_allocated_bytes"] / 1024**2 for j in jobs]),
                "device": row["device"],
                "reduced_nodes": row["synthetic_nodes"],
                "common_gcn_accuracy_mean_percent": row["accuracy_mean"],
                "common_gcn_accuracy_std_percent": row["accuracy_std"],
            }
            for field, summary in (
                ("time_seconds", "reduction_seconds_median"),
                ("peak_process_rss_mib", "peak_rss_mib_median"),
                ("peak_cuda_allocated_mib", "peak_cuda_mib_median"),
            ):
                if metrics[field] is not None:
                    assert math.isclose(metrics[field]["median"], row[summary])
            assert all(j["memory"]["rss_sample_interval_seconds"] == 0.005 for j in jobs)
            pair["implementations"][backend] = metrics
        author, package = (pair["implementations"][b] for b in ("author", "graphslim"))
        assert author["device"] == package["device"]
        assert author["reduced_nodes"] == package["reduced_nodes"]
        author_jobs = [
            j for j in source["jobs"] if (j["method"], j["dataset"]) == (method, dataset)
        ]
        assert all(j["input_sha256"] == author_jobs[0]["input_sha256"] for j in author_jobs)
        pair["author_over_graphslim_time"] = (
            author["time_seconds"]["median"] / package["time_seconds"]["median"]
        )
        pair["common_gcn"] = quality(scores["author"], scores["graphslim"])
        pair["native_gp"] = None
        if method == "gcgp":
            gp = {}
            for backend in ("author", "graphslim"):
                records = [
                    r
                    for r in source["native_gp"]["records"]
                    if (r["dataset"], r["backend"]) == (dataset, backend)
                ]
                assert len(records) == 3
                gp[backend] = statistics.mean(r["test_accuracy"] for r in records)
                summary = next(
                    r
                    for r in source["native_gp"]["summary"]
                    if (r["dataset"], r["backend"]) == (dataset, backend)
                )
                assert math.isclose(gp[backend], summary["mean"])
            pair["native_gp"] = quality(gp["author"], gp["graphslim"])
        pairs.append(pair)
    return {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha(SOURCE),
        "renderer_sha256": sha(__file__),
        "scope": source["scope"],
        "reduction_seeds_per_implementation": 3,
        "uncertainty": "Bars are medians; whiskers show observed min-max across three reductions, not confidence intervals.",
        "memory": "Process RSS sampled every 5 ms includes resident inputs/imports. CUDA is PyTorch peak allocated memory after context warmup; CPU-only cells are not applicable.",
        "fidelity": {
            "criterion": "100 * max(0, author_mean - graphslim_mean) / author_mean < 5",
            "threshold_percent": 5,
            "reference": "Matched pinned author implementation; this is separate from the paper-reference coverage roster.",
            "shared_gcn_passes": sum(
                p["common_gcn"]["relative_shortfall_percent"] < 5 for p in pairs
            ),
            "shared_gcn_pairs": 6,
            "native_gp_passes": sum(
                p["native_gp"]["relative_shortfall_percent"] < 5 for p in pairs if p["native_gp"]
            ),
            "native_gp_pairs": 2,
        },
        "pairs": pairs,
    }


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    author_color, package_color = "#2A78D6", "#EB6834"
    ink, secondary, grid = "#152435", "#536172", "#E3E8EE"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 15,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": secondary,
            "ytick.color": ink,
            "svg.fonttype": "none",
            "svg.hashsalt": "graphslim-controlled-comparison",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.6, 13.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.81, bottom=0.15, wspace=0.39, hspace=0.43)
    fig.patch.set_facecolor("white")
    fig.text(0.055, 0.952, "GraphSlim vs. author implementations", fontsize=28, weight="bold")
    fig.text(
        0.055,
        0.918,
        "Matched inputs and output sizes · same device within each pair · all six comparisons",
        fontsize=16,
        color=secondary,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=author_color,
                hatch="///",
                linewidth=1.3,
                label="Author implementation",
            ),
            Patch(facecolor=package_color, edgecolor=package_color, label="GraphSlim"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.05, 0.893),
        ncol=2,
        frameon=False,
        fontsize=17,
        handlelength=1.6,
        columnspacing=2.6,
    )
    y = np.arange(6)
    labels = [f"{NAMES[p['method']]}\n{p['dataset'].title()}" for p in data["pairs"]]

    def style(ax, title, xlabel):
        ax.set_title(title, loc="left", pad=20)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_yticks(y, labels)
        ax.set_ylim(5.6, -0.65)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.tick_params(axis="x", length=3, color="#B8C1CC")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#A7B2BF")
        for divide in (1.5, 3.5):
            ax.axhline(divide, color=grid, linewidth=0.8)

    specs = [
        (axes[0, 0], "(a) Reduction time", "time_seconds", "Seconds · lower is better", 2.3),
        (
            axes[0, 1],
            "(b) Peak process memory",
            "peak_process_rss_mib",
            "MiB · sampled process RSS",
            2650,
        ),
        (
            axes[1, 0],
            "(c) Peak GPU memory",
            "peak_cuda_allocated_mib",
            "MiB · PyTorch peak allocated",
            345,
        ),
    ]
    for ax, title, field, xlabel, maximum in specs:
        style(ax, title, xlabel)
        ax.set_xlim(0, maximum)
        for backend, offset, color in (
            ("author", -0.19, author_color),
            ("graphslim", 0.19, package_color),
        ):
            for i, pair in enumerate(data["pairs"]):
                stats = pair["implementations"][backend][field]
                if stats is None:
                    if backend == "author":
                        ax.text(
                            maximum * 0.035,
                            i,
                            "N/A · CPU only",
                            va="center",
                            fontsize=14,
                            color=secondary,
                        )
                    continue
                value, low, high = (stats[k] for k in ("median", "minimum", "maximum"))
                ax.barh(
                    i + offset,
                    value,
                    height=0.30,
                    color="white" if backend == "author" else color,
                    edgecolor=color,
                    linewidth=1.1,
                    hatch="///" if backend == "author" else None,
                    xerr=np.array([[value - low], [high - value]]),
                    error_kw=dict(ecolor=ink, elinewidth=0.9, capsize=2.5, capthick=0.9),
                    zorder=3,
                )
                label = f"{value:.3f}" if field == "time_seconds" else f"{value:,.0f}"
                if field == "time_seconds" and backend == "graphslim":
                    label += f"  ({pair['author_over_graphslim_time']:.2f}×)"
                ax.text(
                    high + maximum * 0.025, i + offset, label, va="center", fontsize=13.5, color=ink
                )

    ax = axes[1, 1]
    style(ax, "(d) Accuracy shortfall", "Relative shortfall (%) · lower is better")
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.axvline(5, color=ink, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(5.35, -0.42, "5% criterion", color=secondary, fontsize=13)
    for i, pair in enumerate(data["pairs"]):
        value = pair["common_gcn"]["relative_shortfall_percent"]
        ax.barh(i - 0.18, value, height=0.30, color="#7C8B9D", zorder=3)
        if value == 0:
            ax.plot(0, i - 0.18, marker="|", markersize=12, color="#536172", zorder=4)
        ax.text(
            value + 0.35,
            i - 0.18,
            f"{value:.2f}%" + (" · outside 5%" if value >= 5 else ""),
            va="center",
            fontsize=13.5,
            color=ink,
        )
        if pair["native_gp"]:
            gp = pair["native_gp"]["relative_shortfall_percent"]
            ax.plot(
                gp, i + 0.22, marker="D", markersize=7, color=package_color, clip_on=False, zorder=5
            )
            ax.text(
                gp + 0.65, i + 0.22, f"GP {gp:.2f}%", va="center", fontsize=13.5, color="#A9431E"
            )
    ax.legend(
        handles=[
            Patch(facecolor="#7C8B9D", label="Shared GCN"),
            Line2D(
                [], [], linestyle="none", marker="D", color=package_color, label="Native GP (GCGP)"
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=12.5,
        handlelength=1.2,
        borderaxespad=0,
    )
    footer = fig.text(
        0.055,
        0.062,
        "Time / memory: medians of 3 reductions; whiskers: observed min–max. Accuracy: gap between means (GCN: 30 fits; GP: 3 graphs).",
        fontsize=13,
        color=secondary,
    )
    fig.text(
        0.055,
        0.033,
        "CGC: 4 CPU threads. ClustGDD / GCGP: one RTX PRO 6000 Blackwell. RSS sampled every 5 ms; CUDA is allocated, not reserved memory.",
        fontsize=13,
        color=secondary,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        for text in [ax.title, ax.xaxis.label, *ax.texts, *ax.get_yticklabels()]:
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width, (
                f"Clipped label: {text.get_text()}"
            )
    assert footer.get_window_extent(renderer).y1 < min(
        ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[1]
    ), "Footer overlaps axis labels"
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".svg"), metadata={"Date": None}, facecolor="white")
    svg = STEM.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(STEM.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(
        STEM.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
        facecolor="white",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = reviewed_data()
    if args.check:
        saved = json.loads(DATA.read_text())
        hashes = saved.pop("assets_sha256")
        assert saved == data, "Chart data or renderer changed; regenerate the figure."
        for suffix, digest in hashes.items():
            assert sha(STEM.with_suffix(suffix)) == digest, f"Stale {suffix} figure"
        print("Comparison figure and all six source-backed time/memory/quality pairs match.")
    else:
        plot(data)
        data["assets_sha256"] = {
            ext: sha(STEM.with_suffix(ext)) for ext in (".svg", ".png", ".pdf")
        }
        DATA.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {STEM.relative_to(ROOT)}.svg/.png/.pdf and reviewed chart data.")


if __name__ == "__main__":
    main()

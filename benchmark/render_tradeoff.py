"""Plot reduction time against downstream accuracy for ogbn-arxiv and Reddit.

``--collect RUN_DIR`` freezes complete per-seed records from a dataset-expansion
run into a JSON snapshot. Rendering reads only that snapshot, so the README
figures can be redrawn without the raw run directory.
"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "benchmark" / "results" / "tradeoff_20261028.json"
FIGURES = ROOT / "figures"
DATASETS = {
    "ogbn-arxiv": ("0.001", "0.005", "0.01"),
    "reddit": ("0.0005", "0.001", "0.002"),
}
METHODS = ("cgc", "clustgdd", "gcgp", "random")
LABELS = {"cgc": "CGC", "clustgdd": "ClustGDD", "gcgp": "GCGP", "random": "Random"}
TITLES = {"ogbn-arxiv": "ogbn-arxiv", "reddit": "Reddit"}
MARKERS = {"cgc": "o", "clustgdd": "s", "gcgp": "D", "random": "^"}
SIZES = (48, 90, 150)
# Three validated categorical slots; Random is a baseline and stays neutral gray.
STYLES = {
    "light": dict(
        surface="#fcfcfb",
        primary="#0b0b0b",
        secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series={"cgc": "#2a78d6", "clustgdd": "#eb6834", "gcgp": "#1baf7a", "random": "#898781"},
    ),
    "dark": dict(
        surface="#1a1a19",
        primary="#f0efec",
        secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series={"cgc": "#3987e5", "clustgdd": "#d95926", "gcgp": "#199e70", "random": "#898781"},
    ),
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run_dir):
    """Return a snapshot of complete cells; refuse partial or mixed-source records."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    protocol = manifest["protocol"]
    environments = set()
    cells = []
    for dataset, rates in DATASETS.items():
        groups = [("whole", None, [f"whole__{dataset}"])]
        groups += [
            (
                method,
                rate,
                [f"{dataset}__{method}__r{rate}__s{seed}" for seed in protocol["reduction_seeds"]],
            )
            for method in METHODS
            for rate in rates
        ]
        for method, rate, job_ids in groups:
            records, hashes = [], []
            for job_id in job_ids:
                path = run_dir / "jobs" / job_id / "result.json"
                record = json.loads(path.read_text())
                if record["job"]["id"] != job_id:
                    raise ValueError(f"{path} belongs to {record['job']['id']}")
                if record["source_sha256"] != manifest["source_sha256"]:
                    raise ValueError(f"{job_id} was produced by a different source snapshot")
                if [item["seed"] for item in record["evaluations"]] != protocol["evaluation_seeds"]:
                    raise ValueError(f"{job_id} does not contain every evaluation seed")
                runtime = record["runtime"]
                environments.add((record["gpu_name"], runtime["torch"], runtime["cuda"]))
                records.append(record)
                hashes.append(_sha256(path))
            tests = [item["test"] for record in records for item in record["evaluations"]]
            seconds = [record["timing"]["reduction_seconds"] for record in records]
            nodes = [
                record["reduction"]["synthetic_nodes"]
                if method != "whole"
                else record["data"]["nodes"]
                for record in records
            ]
            cells.append(
                dict(
                    dataset=dataset,
                    method=method,
                    training_node_rate=rate,
                    job_ids=job_ids,
                    result_sha256=hashes,
                    actual_nodes=nodes,
                    n_evaluations=len(tests),
                    test_accuracy_mean=statistics.fmean(tests),
                    test_accuracy_population_std=statistics.pstdev(tests),
                    reduction_seconds=seconds,
                    median_reduction_seconds=statistics.median(seconds),
                )
            )
    if len(environments) != 1:
        raise ValueError(f"Cells mix hardware or runtimes: {sorted(environments)}")
    gpu, torch, cuda = environments.pop()
    try:
        run_label = str(run_dir.relative_to(ROOT))
    except ValueError:
        run_label = str(run_dir)
    return dict(
        schema_version=1,
        snapshot_date=datetime.date.today().isoformat(),
        hardware=dict(gpu=gpu, torch=torch, cuda=cuda),
        protocol=protocol,
        source=dict(
            run_dir=run_label,
            source_sha256=manifest["source_sha256"],
            archive_sha256=manifest["archive_sha256"],
            manifest_sha256=_sha256(run_dir / "manifest.json"),
        ),
        statistics=dict(
            accuracy=(
                "mean and population standard deviation over 3 reduction seeds x 10 GCN "
                "seeds per reduced cell; the full graph has 10 GCN seeds"
            ),
            reduction_time=(
                "median of the three timing.reduction_seconds values; includes the reducer "
                "call and reduced-graph saving, excludes data loading and GCN fits"
            ),
        ),
        notes=dict(
            reddit="The package loader uses the PyG Reddit2 graph.",
            nope="NOPE reached the 3600 s job limit on both datasets and has no cell.",
        ),
        cells=cells,
    )


def _rate_label(rate):
    return f"{float(rate) * 100:g}%"


def _upper(value, step):
    return step * math.ceil(value / step)


def render(snapshot, output_dir):
    """Write one light and one dark PNG per dataset; return the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, rates in DATASETS.items():
        cells = [cell for cell in snapshot["cells"] if cell["dataset"] == dataset]
        whole = next(cell for cell in cells if cell["method"] == "whole")
        reduced = [cell for cell in cells if cell["method"] != "whole"]
        times = [cell["median_reduction_seconds"] for cell in reduced]
        top = max(
            [100 * whole["test_accuracy_mean"]]
            + [
                100 * (cell["test_accuracy_mean"] + cell["test_accuracy_population_std"])
                for cell in reduced
            ]
        )
        xlim = (10 ** math.floor(math.log10(min(times) / 1.3)), max(times) * 4)
        ylim = (0, min(100, _upper(top + 4, 10)))
        for mode, style in STYLES.items():
            plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
            fig, ax = plt.subplots(figsize=(4.8, 3.7), dpi=200)
            fig.patch.set_facecolor(style["surface"])
            ax.set_facecolor(style["surface"])
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            decades = range(math.floor(math.log10(xlim[0])), math.ceil(math.log10(xlim[1])) + 1)
            ax.xaxis.set_major_locator(FixedLocator([10.0**power for power in decades]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.grid(True, color=style["grid"], linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(style["axis"])
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=style["muted"], labelcolor=style["secondary"], length=0, pad=4)
            ax.set_xlabel("Median graph reduction time (s, log scale)", color=style["secondary"])
            ax.set_ylabel("Test accuracy (%)", color=style["secondary"])
            ax.set_title(
                TITLES[dataset],
                loc="left",
                color=style["primary"],
                fontsize=12,
                fontweight="bold",
                pad=10,
            )

            full = 100 * whole["test_accuracy_mean"]
            ax.axhline(full, color=style["secondary"], linewidth=1.0, zorder=1)
            ax.text(
                xlim[1] / 1.08,
                full + 0.012 * (ylim[1] - ylim[0]),
                f"Full graph {full:.1f}%",
                ha="right",
                va="bottom",
                color=style["secondary"],
                fontsize=8.5,
            )

            for method in METHODS:
                color = style["series"][method]
                rows = [
                    next(
                        cell
                        for cell in reduced
                        if cell["method"] == method and cell["training_node_rate"] == rate
                    )
                    for rate in rates
                ]
                xs = [cell["median_reduction_seconds"] for cell in rows]
                ys = [100 * cell["test_accuracy_mean"] for cell in rows]
                errors = [100 * cell["test_accuracy_population_std"] for cell in rows]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=errors,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    alpha=0.75,
                    zorder=2,
                )
                # Smaller budgets draw last so large markers cannot hide them.
                for index, (x, y, size) in enumerate(zip(xs, ys, SIZES)):
                    ax.scatter(
                        [x],
                        [y],
                        s=size,
                        marker=MARKERS[method],
                        color=color,
                        edgecolors=style["surface"],
                        linewidths=1.2,
                        zorder=3 + (len(SIZES) - index) / 10,
                    )

            method_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=MARKERS[method],
                    markersize=7,
                    markerfacecolor=style["series"][method],
                    markeredgecolor=style["surface"],
                    label=LABELS[method],
                )
                for method in METHODS
            ]
            rate_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=math.sqrt(size),
                    markerfacecolor="none",
                    markeredgecolor=style["secondary"],
                    label=_rate_label(rate),
                )
                for size, rate in zip(SIZES, rates)
            ]
            legend_style = dict(
                frameon=True,
                facecolor=style["surface"],
                edgecolor=style["grid"],
                labelcolor=style["primary"],
                fontsize=8.5,
                title_fontsize=8.5,
                borderpad=0.7,
                handletextpad=0.5,
            )
            methods_legend = ax.legend(
                handles=method_handles, title="Method", loc="lower right", **legend_style
            )
            methods_legend.get_title().set_color(style["secondary"])
            ax.add_artist(methods_legend)
            rate_legend = ax.legend(
                handles=rate_handles,
                title="Budget",
                loc="lower left",
                labelspacing=0.9,
                **legend_style,
            )
            rate_legend.get_title().set_color(style["secondary"])

            fig.tight_layout()
            path = output_dir / f"tradeoff_{dataset}_{mode}.png"
            fig.savefig(path, facecolor=style["surface"])
            plt.close(fig)
            written.append(path)
    return written


TABLE_TEXT = {
    "en": dict(
        header=(
            "Dataset",
            "Method",
            "Budget",
            "Reduced nodes",
            "Test accuracy (%)",
            "Reduction time (s)",
        ),
        whole="Full graph",
    ),
    "zh": dict(
        header=("数据集", "方法", "预算", "缩减后节点数", "测试准确率（%）", "图缩减耗时（s）"),
        whole="原图",
    ),
}


def markdown_table(snapshot, language="en"):
    """Return the README table of plotted values in the requested language."""
    text = TABLE_TEXT[language]
    lines = [
        "| " + " | ".join(text["header"]) + " |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in snapshot["cells"]:
        whole = cell["method"] == "whole"
        nodes = sorted(set(cell["actual_nodes"]))
        mean = 100 * cell["test_accuracy_mean"]
        std = 100 * cell["test_accuracy_population_std"]
        row = (
            TITLES[cell["dataset"]],
            text["whole"] if whole else LABELS[cell["method"]],
            "—" if whole else _rate_label(cell["training_node_rate"]),
            "/".join(f"{value:,}" for value in nodes),
            f"{mean:.2f} ± {std:.2f}",
            "—" if whole else f"{cell['median_reduction_seconds']:.2f}",
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", type=Path, metavar="RUN_DIR")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--table", choices=sorted(TABLE_TEXT), help="print the README table")
    options = parser.parse_args()
    if options.collect:
        snapshot = collect(options.collect)
        options.snapshot.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"Wrote {options.snapshot}")
    snapshot = json.loads(options.snapshot.read_text())
    if options.table:
        print(markdown_table(snapshot, options.table))
        return
    for path in render(snapshot, options.output_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

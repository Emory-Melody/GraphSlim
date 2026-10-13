"""Plan or run benchmark jobs serially using the shared package commands."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from graphslim.reduction import get_method_spec

# Fractions of labeled training nodes, as expected by GraphSlim's CLI.
# These differ from the whole-graph percentages printed for transductive data.
RATES = {
    "citeseer": ("0.1", "0.25", "0.5"),
    "cora": ("0.1", "0.25", "0.5"),
    "pubmed": ("0.05", "0.1", "0.5"),
    "ogbn-arxiv": ("0.001", "0.005", "0.01"),
    "flickr": ("0.001", "0.005", "0.01"),
    "reddit": ("0.0005", "0.001", "0.002"),
    "yelp": ("0.0005", "0.001", "0.002"),
}
PERFORMANCE_METHODS = (
    "cent_d",
    "cent_p",
    "random",
    "herding",
    "kcenter",
    "t_spanner",
    "averaging",
    "variation_neighborhoods",
    "vng",
    "clustering",
    "cgc",
    "clustgdd",
    "gcgp",
    "nope",
    "geom",
    "sfgc",
    "gcsntk",
    "gcdmx",
    "gcondx",
    "gcond",
    "doscond",
    "msgc",
    "sgdd",
    "gdem",
)
SCALABILITY_RATES = {
    "ogbn-arxiv": ("0.00045", "0.0005", "0.001", "0.005", "0.01", "0.025", "0.05", "0.1"),
    "reddit": (
        "0.00025",
        "0.0005",
        "0.001",
        "0.002",
        "0.005",
        "0.01",
        "0.015",
        "0.02",
        "0.025",
        "0.03",
    ),
}
SUITES = (
    "performance",
    "scalability",
    "data_initialization",
    "transferability",
    "nas",
    "graph_property_preservation",
    "robustness",
)


@dataclass(frozen=True)
class Job:
    module: str
    arguments: tuple

    def command(self, python=sys.executable):
        return [
            python,
            "-m",
            self.module,
            "--load_path",
            str(ROOT / "data"),
            "--save_path",
            str(ROOT / "checkpoints"),
            *self.arguments,
        ]


def plan(suite, datasets=None, methods=None, *, whole=True):
    """Build the experiment matrix without importing Torch or loading datasets."""
    default_datasets = {
        "performance": tuple(RATES),
        "scalability": tuple(SCALABILITY_RATES),
        "data_initialization": ("cora", "ogbn-arxiv", "reddit"),
        "transferability": ("cora", "ogbn-arxiv", "reddit"),
        "nas": ("cora", "ogbn-arxiv", "reddit"),
        "graph_property_preservation": ("cora", "citeseer", "ogbn-arxiv", "flickr", "reddit"),
        "robustness": ("cora", "citeseer", "flickr"),
    }
    default_methods = {
        "performance": PERFORMANCE_METHODS,
        "scalability": ("random", "herding", "kcenter", "gcondx", "geom", "gcond", "msgc"),
        "data_initialization": ("gcondx", "geom", "doscond", "gcond", "msgc"),
        "transferability": (
            "random",
            "kcenter",
            "averaging",
            "vng",
            "gcondx",
            "geom",
            "sfgc",
            "doscond",
            "gcond",
            "msgc",
            "sgdd",
        ),
        "nas": ("random", "kcenter", "gcondx", "geom", "sfgc", "doscond", "gcond", "msgc"),
        "robustness": ("random", "kcenter", "gcondx", "geom", "gcond"),
        "graph_property_preservation": ("vng", "gcond", "msgc", "sgdd"),
    }
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    datasets = tuple(datasets or default_datasets[suite])
    methods = tuple(
        dict.fromkeys(get_method_spec(m).name for m in (methods or default_methods[suite]))
    )
    jobs = []
    for dataset in datasets:
        if suite == "graph_property_preservation":
            for module in ("benchmark.graph_property", "benchmark.graph_property_no_structure"):
                if whole:
                    jobs.append(Job(module, ("-D", dataset, "-W")))
                for method in methods:
                    jobs.append(Job(module, ("-D", dataset, "-M", method)))
            continue
        for method in methods:
            base = ("-D", dataset, "-M", method)
            if suite in ("performance", "scalability"):
                rates = (RATES if suite == "performance" else SCALABILITY_RATES)[dataset]
                for rate in rates:
                    jobs.append(Job("graphslim.train_all", base + ("-R", rate)))
            elif suite == "data_initialization":
                for init in ("random", "averaging", "kcenter", "herding", "clustering"):
                    jobs.append(Job("graphslim.train_all", base + ("--init", init)))
            elif suite == "robustness":
                for attack in ("metattack", "random_adj", "random_feat"):
                    jobs.append(Job("graphslim.train_all", base + ("-A", attack, "-P", "0.5")))
            else:
                module = (
                    "benchmark.run_cross_arch"
                    if suite == "transferability"
                    else "benchmark.run_nas"
                )
                jobs.append(Job(module, base))
        if whole and suite == "performance":
            jobs.append(Job("graphslim.run_eval", ("-D", dataset, "-W")))
        if whole and suite == "robustness":
            for attack in ("metattack", "random_adj", "random_feat"):
                jobs.append(
                    Job("graphslim.run_eval", ("-D", dataset, "-W", "-A", attack, "-P", "0.5"))
                )
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--datasets", nargs="+", choices=tuple(RATES))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="print commands without training")
    parser.add_argument("--skip-whole", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--slurm", action="store_true", help="submit each job with sbatch --wrap")
    parser.add_argument(
        "--slurm-arg", action="append", default=[], help="sbatch option; repeat as needed"
    )
    options, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    try:
        jobs = plan(options.suite, options.datasets, options.methods, whole=not options.skip_whole)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    for job in jobs:
        command = job.command(options.python) + extra
        if options.slurm:
            command = ["sbatch", *options.slurm_arg, "--wrap", shlex.join(command)]
        print(shlex.join(command), flush=True)
        if not options.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    status = "planned" if options.dry_run else ("submitted" if options.slurm else "completed")
    print(f"{len(jobs)} jobs {status}.", flush=True)


if __name__ == "__main__":
    main()

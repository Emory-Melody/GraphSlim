"""Render the frozen published results or check the README for drift."""

import argparse
import csv
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results" / "gc4nc_table9.csv"
START = "<!-- benchmark-results:start -->"
END = "<!-- benchmark-results:end -->"
GROUPS = {
    "Coreset / sparsification": ("Cent-D", "Cent-P", "Random", "Herding", "K-Center", "TSpanner"),
    "Coarsening": ("Averaging", "VN", "VNG", "Clustering"),
    "Condensation": (
        "GEOM",
        "SFGC",
        "GCSNTK",
        "GCDMX",
        "GCondX",
        "GCond",
        "DosCond",
        "MSGC",
        "SGDD",
        "GDEM",
        "DosCondX",
        "GCDM",
        "SimGC",
        "GECC",
    ),
    "Baseline": ("Whole",),
}


def read_results(path=RESULTS):
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    keys = [(row["Dataset"], row["Reduction rate (%)"]) for row in rows]
    if len(keys) != len(set(keys)) or len(rows) != 21:
        raise ValueError("Expected 21 unique dataset/rate rows")
    for row in rows:
        for methods in GROUPS.values():
            for method in methods:
                value = row[method]
                if value in ("NR", "-", "OOM"):
                    continue
                if not re.fullmatch(r"\d+(?:\.\d+)?(?:±\d+(?:\.\d+)?)?", value):
                    raise ValueError(f"Invalid published value: {value!r}")
                if not 0 <= float(value.split("±")[0]) <= 100:
                    raise ValueError(f"Out-of-range percentage: {value!r}")
    return rows


def render_results(rows=None):
    rows = read_results() if rows is None else rows
    sections = []
    for dataset in dict.fromkeys(row["Dataset"] for row in rows):
        selected = [row for row in rows if row["Dataset"] == dataset]
        if len(selected) != 3:
            raise ValueError(f"Expected three rates for {dataset}")
        metric = "F1-macro (%)" if dataset == "Yelp" else "test accuracy (%)"
        sections.append(f"### {dataset} — {metric}\n")
        headers = ["Family", "Method", *[row["Reduction rate (%)"] + "%" for row in selected]]
        sections.append("| " + " | ".join(headers) + " |")
        sections.append("| --- | --- | ---: | ---: | ---: |")
        for family, methods in GROUPS.items():
            for method in methods:
                values = [row[method].replace("±", " ± ") for row in selected]
                sections.append("| " + " | ".join([family, method, *values]) + " |")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the embedded README tables")
    options = parser.parse_args(argv)
    rendered = render_results()
    if options.check:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        if readme.count(START) != 1 or readme.count(END) != 1:
            raise SystemExit("README results markers are missing or duplicated")
        embedded = readme.split(START, 1)[1].split(END, 1)[0].strip()
        if embedded != rendered.strip():
            raise SystemExit("README results differ from the frozen CSV")
        print("README matches all 21 published dataset/rate rows.")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()

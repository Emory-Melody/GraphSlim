"""Publish complete package NR cells from explicitly scoped immutable run batches.

This is a provenance-aware coverage table, not a controlled cross-batch ranking.
Published measurements and historical tables remain separate and unchanged.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics
import zipfile

from benchmark import rerun_nr as nr
from benchmark.reproduce import RATES

ROOT = nr.ROOT
CATALOG = ROOT / "benchmark/results/package_runs.json"
START = "<!-- package-results:start -->"
END = "<!-- package-results:end -->"


def verify_archive(folder, expected):
    digest = hashlib.sha256()
    path = folder / "source.zip"
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                raise RuntimeError("Unexpected directory in frozen scientific source")
            digest.update(item.filename.encode())
            digest.update(archive.read(item))
    if digest.hexdigest() != expected:
        raise RuntimeError("Frozen source archive differs from manifest")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_cell(folder, manifest, entry, method, dataset, rate):
    if entry["format"] not in ("hpo", "nr"):
        raise RuntimeError("Unknown run record format")
    hpo = entry["format"] == "hpo"
    jobs = [
        job
        for job in manifest["jobs"]
        if (job["method"], job["dataset"], job["rate"]) == (method, dataset, rate)
        and (not hpo or job["kind"] == "final")
    ]
    if len(jobs) != 3 or {job["seed"] for job in jobs} != set(nr.SEEDS):
        raise RuntimeError(f"Incomplete/duplicate roster: {method}/{dataset}/{rate}")
    cell = {
        "method": method,
        "dataset": dataset,
        "rate": rate,
        "run": entry["path"],
        "source_sha256": manifest["source_sha256"],
        "metric": "F1-macro" if dataset == "yelp" else "accuracy",
        "status": "pending",
        "value": "PENDING (0/3)",
        "n": 0,
    }
    paths = [folder / "jobs" / job["id"] / "result.json" for job in jobs]
    done = sum(path.is_file() for path in paths)
    cell["value"] = f"PENDING ({done}/3)"
    if not all(path.is_file() for path in paths):
        if any((path.parent / "error.json").is_file() for path in paths):
            cell.update(status="failed", value=f"FAILED ({done}/3)")
        elif any(job.get("blocked_reason") for job in jobs):
            cell.update(status="blocked", value="CONFIG")
        return cell
    records = [nr.read_json(path) for path in paths]
    for job, record in zip(jobs, records):
        if record["job"] != job or record["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError("Raw result job/source differs from frozen manifest")
        evaluations = record["evaluations"]
        expected_seeds = set(range(10)) if hpo else set(range(job["seed"], job["seed"] + 10))
        if len(evaluations) != 10 or {item["seed"] for item in evaluations} != expected_seeds:
            raise RuntimeError("Missing/duplicate evaluation seeds")
        if any(
            not math.isfinite(item[key]) or not 0 <= item[key] <= 1
            for item in evaluations
            for key in ("validation", "test")
        ):
            raise RuntimeError("Invalid raw evaluation metrics")
    if any(record["data"] != records[0]["data"] for record in records):
        raise RuntimeError("Mixed datasets/splits within a cell")
    if hpo:
        selection_path = folder / "selected" / f"{method}__{dataset}.json"
        selection = nr.read_json(selection_path)
        digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if (
            selection["source_sha256"] != manifest["source_sha256"]
            or selection["data"] != records[0]["data"]
        ):
            raise RuntimeError("Selection source/data differs from final records")
        if any(record["selection_sha256"] != digest for record in records):
            raise RuntimeError("Mixed/tampered final selection")
        cell["selection_sha256"] = digest
    scores = [item["test"] for record in records for item in record["evaluations"]]
    mean, std = statistics.mean(scores), statistics.pstdev(scores)
    cell.update(
        status="complete",
        value=f"{100 * mean:.2f} ± {100 * std:.2f}",
        mean=mean,
        std=std,
        n=len(scores),
        data=records[0]["data"],
        result_sha256={
            str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    )
    return cell


def collect(catalog=CATALOG, root=ROOT):
    entries = nr.read_json(catalog)["runs"]
    cells = {}
    sources = []
    for entry in entries:
        folder = root / entry["path"]
        manifest_path = folder / "manifest.json"
        manifest = nr.read_json(manifest_path) if manifest_path.is_file() else None
        sources.append(
            {
                **entry,
                "source_sha256": manifest["source_sha256"] if manifest else None,
                "archive_sha256": verify_archive(folder, manifest["source_sha256"])
                if manifest
                else None,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest
                else None,
            }
        )
        for method in entry["methods"]:
            for dataset in entry["datasets"]:
                for rate in RATES[dataset]:
                    key = (method, dataset, rate)
                    if manifest is None:
                        cells[key] = {
                            "method": method,
                            "dataset": dataset,
                            "rate": rate,
                            "run": entry["path"],
                            "status": "not started",
                            "n": 0,
                            "value": "PENDING (0/3)",
                            "metric": "F1-macro" if dataset == "yelp" else "accuracy",
                            "source_sha256": None,
                        }
                    else:
                        cells[key] = collect_cell(folder, manifest, entry, *key)
    expected = {
        (method, dataset, rate)
        for method in nr.METHODS
        for dataset in RATES
        for rate in RATES[dataset]
    }
    if set(cells) != expected:
        raise RuntimeError("Package table must cover exactly 105 method/dataset/rate cells")
    return list(cells.values()), sources


def render(cells, sources):
    completed = sum(cell["status"] == "complete" for cell in cells)
    lines = [
        "## Package measurements completing published NR coverage",
        "",
        f"Complete local cells: {completed}/105 (7 datasets × 3 CLI rates × 5 methods).",
        "Only cells with all 3 reduction seeds × 10 downstream test fits are numeric.",
        "Values are test mean ± population standard deviation (%), not the published",
        "table's reported standard errors. Yelp uses F1-macro; all other rows use accuracy.",
        "Pending repaired SimGC cells do not fall back to superseded measurements.",
        "",
        "This is an explicitly source-scoped coverage table, not a controlled ranking:",
        "NR methods use the archived fixed-GCN protocol; GECC–Yelp uses paper-grid HPO;",
        "SimGC uses a fresh repaired source and validation-only GCN HPO (citation raw",
        "features/no-activation/3,001 updates; other datasets 300 updates). Source",
        "code, preprocessing, training budgets and evaluation seed schedules differ.",
        "Published results and earlier local batches remain separately preserved above.",
        "",
        "| Dataset | CLI rate | Clustering | DosCondX | GCDM | SimGC (repaired HPO) | GECC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    indexed = {(cell["method"], cell["dataset"], cell["rate"]): cell for cell in cells}
    for dataset in RATES:
        for rate in RATES[dataset]:
            values = [indexed[(method, dataset, rate)]["value"] for method in nr.METHODS]
            lines.append("| " + " | ".join([dataset, rate, *values]) + " |")
    lines += ["", "Source assignment (no scores pooled across batches):", ""]
    for source in sources:
        lines.append(
            f"- `{', '.join(source['methods'])}` on `{', '.join(source['datasets'])}`: "
            f"`{source['path']}` — {source['label']}."
        )
    lines += [
        "",
        "[Machine-readable coverage and per-cell provenance](benchmark/results/package_local_results.json);",
        "[CSV export](benchmark/results/package_local_results.csv). `PENDING`/`FAILED`/`CONFIG`",
        "are unavailable results, not zero. Published `NR` means absent from the paper,",
        "not absent from this local coverage table. Raw artifacts live in the listed run directories.",
        "",
    ]
    return "\n".join(lines)


def export(update_readme=False, check=False):
    cells, sources = collect()
    rendered = render(cells, sources)
    path = ROOT / "README.md"
    text = path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise RuntimeError("Package README markers missing/duplicated")
    before, rest = text.split(START)
    embedded, after = rest.split(END)
    if check:
        if embedded.strip() != rendered.strip():
            raise RuntimeError("Package README differs from verified raw records")
        return cells
    nr.write_json(
        ROOT / "benchmark/results/package_local_results.json",
        {"utc": nr.utcnow(), "sources": sources, "cells": cells},
    )
    fields = (
        "dataset",
        "rate",
        "method",
        "metric",
        "status",
        "value",
        "n",
        "mean",
        "std",
        "run",
        "source_sha256",
    )
    with (ROOT / "benchmark/results/package_local_results.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)
    if update_readme:
        path.write_text(before + START + "\n\n" + rendered + "\n" + END + after)
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args(argv)
    cells = export(options.update_readme, options.check)
    print(f"Verified complete cells: {sum(cell['status'] == 'complete' for cell in cells)}/105")


if __name__ == "__main__":
    main()

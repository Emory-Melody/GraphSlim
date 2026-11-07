#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Recalculate the 110-pair release results using only Python's standard library.

The unchanged 1.3.1 artifact is embedded in ``base_20261105``. Its own verifier first
checks every file hash and recomputes its 124-pair results. This verifier then drops
the methods listed in ``exclusions.json``, substitutes the rerun scores listed in
``replacements.json`` (recomputed here from their stored test scores), and recomputes
every count. ``rerun_results.json`` records fresh scores of the excluded methods for
transparency; it never enters a count.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics

THRESHOLDS = (1, 2, 5, 10)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def base_module(base):
    spec = importlib.util.spec_from_file_location(
        "base_verify", base / "base_20261105" / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate(base):
    parent = base_module(base)
    root = base / "base_20261105"
    parent.verify(root)
    exclusions = load(base / "exclusions.json")
    excluded = set(exclusions["methods"])
    assert excluded == {"cent_p", "gdem"} and set(exclusions["reasons"]) == excluded
    rows = load(root / "settings.json")
    with (root / "pairs.csv").open(newline="") as f:
        base_pairs = list(csv.DictReader(f))
    assert len(base_pairs) == 124
    kept_keys = [(p["method"], p["dataset"]) for p in base_pairs if p["method"] not in excluded]
    replacements = load(base / "replacements.json")
    substitute = {}
    for record in replacements["records"]:
        key = (record["method"], record["dataset"], record["setting"])
        assert record["method"] not in excluded and key not in substitute
        assert len(record["tests"]) == 30
        value = 100 * statistics.mean(record["tests"])
        assert abs(value - record["reproduced_percent"]) < 1e-9, key
        substitute[key] = value
    grouped = {}
    for row in rows:
        if row["method"] not in excluded:
            key = (row["method"], row["dataset"], row["setting"])
            if key in substitute:
                row = dict(row, reproduced_percent=substitute.pop(key))
            grouped.setdefault((row["method"], row["dataset"]), []).append(row)
    assert not substitute, "Replacement without a matching base setting"
    rerun = {(r["method"], r["dataset"]) for r in replacements["records"]}
    assert sorted(grouped) == sorted(kept_keys)
    derived, uncorrected = [], []
    for key in kept_keys:
        settings = grouped[key]
        values = [parent.errors(r["reference_percent"], r["reproduced_percent"]) for r in settings]
        raw = [
            parent.errors(
                (
                    r["reference_correction"]["table9"]
                    if r["reference_correction"]
                    else r["reference_percent"]
                ),
                r["reproduced_percent"],
            )
            for r in settings
        ]
        pair = dict(
            method=key[0],
            dataset=key[1],
            complete_result=True,
            credited_by_policy=False,
            settings=len(settings),
            reference_source=settings[0]["reference_source"],
            shared_observations_with_reference=all(
                r["shared_observations_with_reference"] for r in settings
            ),
            rerun_after_fix=key in rerun,
        )
        for i, metric in enumerate(parent.METRICS):
            pair[metric] = statistics.mean(v[i] for v in values)
        derived.append(pair)
        uncorrected.append(
            dict(
                pair,
                **{
                    metric: statistics.mean(v[i] for v in raw)
                    for i, metric in enumerate(parent.METRICS)
                },
            )
        )
    independent = [p for p in derived if not p["shared_observations_with_reference"]]
    summary = dict(
        artifact_id="graphslim-reproducibility-20261106",
        base_artifact="graphslim-reproducibility-20261105",
        excluded_methods=sorted(excluded),
        replaced_settings=len(replacements["records"]),
        setting_count=sum(len(v) for v in grouped.values()),
        pair_count=len(derived),
        method_count=len({p["method"] for p in derived}),
        dataset_count=len({p["dataset"] for p in derived}),
        threshold_rule="strict < after averaging per-setting errors within a pair",
        shared_observation_pairs=len(derived) - len(independent),
        cohort_results={
            f"paper_{len(derived)}": parent.aggregate(derived),
            f"without_shared_observations_{len(independent)}": parent.aggregate(independent),
        },
        uncorrected_table9_sensitivity=parent.aggregate(uncorrected),
    )
    assert (summary["pair_count"], summary["method_count"], summary["dataset_count"]) == (
        110,
        17,
        7,
    )
    reruns = load(base / "rerun_results.json")["records"]
    assert {(r["method"], r["dataset"], r["setting"]) for r in reruns} == {
        (r["method"], r["dataset"], r["setting"]) for r in rows if r["method"] in excluded
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(derived[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(derived)
    return summary, stream.getvalue()


def files(base):
    return {
        str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*"))
        if p.is_file()
        and p.name != "manifest.json"
        and not str(p.relative_to(base)).startswith("base_20261105")
        and "__pycache__" not in p.parts
    }


def verify(base):
    manifest = load(base / "manifest.json")
    assert manifest["files_sha256"] == files(base), "File hashes differ from manifest"
    summary, pairs = calculate(base)
    assert summary == load(base / "summary.json"), (
        "Summary differs from setting-level recalculation"
    )
    assert pairs == (base / "pairs.csv").read_text(), "Pair-level metrics differ"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify hashes and recompute every metric (default)."
    )
    parser.add_argument(
        "--write", action="store_true", help="Regenerate derived tables and manifest."
    )
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if args.write:
        summary, pairs = calculate(base)
        (base / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (base / "pairs.csv").write_text(pairs)
        manifest = dict(artifact_id=summary["artifact_id"], files_sha256=files(base))
        (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = verify(base)
    print(
        f"Verified {summary['setting_count']} settings, {summary['pair_count']} pairs, "
        f"{summary['method_count']} methods and {summary['dataset_count']} datasets "
        f"(excluded: {', '.join(summary['excluded_methods'])}; "
        f"{summary['replaced_settings']} settings rerun after fixes)."
    )
    names = {
        "mean_shortfall_percent": "one-sided shortfall",
        "mean_relative_gap_percent": "two-sided error",
    }
    for cohort, results in [
        *summary["cohort_results"].items(),
        ("uncorrected_table9_sensitivity", summary["uncorrected_table9_sensitivity"]),
    ]:
        for metric, label in names.items():
            values = ", ".join(
                f"<{t}% {v['count']}/{v['denominator']} ({100 * v['count'] / v['denominator']:.2f}%)"
                for t, v in results[metric].items()
            )
            print(f"{cohort} / {label}: {values}")


if __name__ == "__main__":
    main()

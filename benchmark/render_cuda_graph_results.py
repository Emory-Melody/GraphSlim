"""Summarize complete or explicitly partial CUDA integration artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from graphslim.acceleration import method_acceleration_support
from graphslim.models import DOWNSTREAM_MODELS
from graphslim.reduction import list_methods


def validated_reductions(path):
    """Publish frozen paired measurements without dropping traces or provenance."""
    payload = path.read_bytes()
    record = json.loads(payload)
    if record.get("status") != "complete" or not record.get("cases"):
        raise ValueError("Reduction measurements must be complete")
    keys = set()
    for case in record["cases"]:
        key = (case["method"], case["arithmetic"], case["seed"])
        if key in keys:
            raise ValueError("Duplicate reduction pair")
        keys.add(key)
        if case["status"] != "pass" or not case["tensors_equal"] or not case["losses_equal"]:
            raise ValueError("Cannot publish a failed pair as exact reduction timing")
        eager, graphed = case["eager"], case["cuda_graph"]
        if eager["hashes"] != graphed["hashes"]:
            raise ValueError("Reduction tensor hashes differ")
        if eager.get("GP_loss_trace") != graphed.get("GP_loss_trace"):
            raise ValueError("GP loss histories differ")
        traces = lambda item: [
            stage["loss_trace"] for stage in item["stages"] if "loss_trace" in stage
        ]
        if traces(eager) != traces(graphed):
            raise ValueError("Training loss histories differ")
        left = eager["reduction_seconds_including_setup"]
        right = graphed["reduction_seconds_including_setup"]
        if not all(math.isfinite(value) and value > 0 for value in (left, right)):
            raise ValueError("Reduction timing must be positive and finite")
        if not math.isclose(left / right, case["speedup_including_setup"], rel_tol=1e-12):
            raise ValueError("Reduction speed ratio does not match its timings")
    return {"raw_result_sha256": hashlib.sha256(payload).hexdigest(), "record": record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reducer-results", type=Path)
    options = parser.parse_args()
    manifest = json.loads((options.run_dir / "manifest.json").read_text())
    state = json.loads((options.run_dir / "state.json").read_text())
    groups = []
    raw_hashes = {}
    for name in [
        "all_methods_gpu_0",
        "all_methods_gpu_1",
        "all_methods_gpu_2",
        "noise_trans",
        "noise_ind",
    ]:
        path = options.run_dir / f"{name}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        groups.append(
            {
                "name": name,
                "status": raw["status"],
                "scope": raw["scope"],
                "expected": raw["expected_cases"],
                "cases": raw["cases"],
            }
        )
        raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    clean = [
        case
        for group in groups
        if group["name"].startswith("all_methods")
        for case in group["cases"]
    ]
    keys = [(case["method"], case["model"]) for case in clean]
    assert len(keys) == len(set(keys)), "Duplicate method/model rows"
    if state.get("status") == "complete":
        assert set(keys) == {
            (method, model) for method in list_methods() for model in DOWNSTREAM_MODELS
        }
        assert all(group["status"] == "complete" for group in groups)
    summary = []
    for model in DOWNSTREAM_MODELS:
        cases = [case for case in clean if case["model"] == model]
        measured = [case for case in cases if case["status"] == "pass"]
        captured = [
            case
            for case in measured
            if any(stage["status"] == "captured" for stage in case["cuda_graph"]["stages"])
        ]
        summary.append(
            {
                "model": model,
                "covered_method_tags": len(cases),
                "exact_pairs": len(measured),
                "captured_pairs": len(captured),
                "eager_pairs": len(measured) - len(captured),
                "median_fit_speedup_including_setup": statistics.median(
                    case["fit_speedup_including_setup"] for case in captured
                )
                if captured
                else None,
                "captured_speedup_range_including_setup": [
                    min(case["fit_speedup_including_setup"] for case in captured),
                    max(case["fit_speedup_including_setup"] for case in captured),
                ]
                if captured
                else None,
                "scope": "80-step synthetic-graph fitting, deterministic Torch/CuBLAS audit, includes setup, validation and loss tracing; not reducer or dataset speedup.",
            }
        )
    result = {
        "scope": "All-method-tag CUDA integration audit; common synthetic graphs, not actual outputs of every reducer or scientific accuracy results.",
        "state": state,
        "source_manifest": manifest,
        "source_archive_sha256": manifest["source_archive_sha256"],
        "raw_result_sha256": raw_hashes,
        "method_policy": method_acceleration_support(),
        "default_enabled": False,
        "model_summary": summary,
        "clean_pairs": len(clean),
        "expected_clean_pairs": 280,
        "noise_pairs": sum(
            len(group["cases"]) for group in groups if group["name"].startswith("noise")
        ),
        "expected_noise_pairs": 336,
        "failed_or_different_cases": [
            {key: case.get(key) for key in ("method", "model", "noise", "status", "error")}
            for group in groups
            for case in group["cases"]
            if case["status"] != "pass"
        ],
    }
    if options.reducer_results is not None:
        result["reduction_pairs"] = validated_reductions(options.reducer_results)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "clean_pairs",
                    "noise_pairs",
                    "model_summary",
                    "failed_or_different_cases",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check complete current-version CPU/GPU demo records without retraining."""

import hashlib
import json
from pathlib import Path
import statistics


def verify(folder):
    summary = json.loads((folder / "demo-summary.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "complete" and status["exit_code"] == 0
    assert provenance["source_commit"] == "ecbbd608a4c915359d8c99b38fe875eb614aa8ce"
    assert provenance["source_version"] == "1.3.4"
    assert provenance["packages"]["graphslim"] == "1.3.4"
    assert provenance["torch_threads"] == 4
    assert summary["config"]["epochs"] == 1000
    assert summary["config"]["outer_loop"] == 20
    assert summary["config"]["inner_loop"] == 15
    assert summary["config"]["eval_epochs"] == 300
    assert summary["config"]["run_eval"] == 10
    assert summary["config"]["config"] is None
    assert summary["config"]["seed"] == 1
    assert summary["shapes"] == {"feat_syn": [70, 1433], "adj_syn": [70, 70], "labels_syn": [70]}
    assert [r["seed"] for r in summary["fits"]] == list(range(1, 11))
    values = [r["test"] for r in summary["fits"]]
    assert all(0 <= r["test"] <= 1 and 0 <= r["validation"] <= 1 for r in summary["fits"])
    assert abs(statistics.mean(values) - summary["test_mean"]) < 1e-12
    assert abs(statistics.pstdev(values) - summary["test_population_std"]) < 1e-12
    return summary, provenance, status


def main():
    root = Path(__file__).resolve().parent
    runs = root / "runs" if (root / "runs").exists() else root
    records = {device: verify(runs / device) for device in ("cpu", "gpu")}
    a, b = records["cpu"][1], records["gpu"][1]
    for key in ("source_commit", "source_sha256", "input_files_sha256", "packages", "torch_threads"):
        assert a[key] == b[key], key
    ignored = {"device", "gpu_id", "save_path"}
    ca, cb = records["cpu"][0]["config"], records["gpu"][0]["config"]
    assert {k: v for k, v in ca.items() if k not in ignored} == {k: v for k, v in cb.items() if k not in ignored}
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["files_sha256"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    for device, (s, _, status) in records.items():
        print(f"{device}: {100 * s['test_mean']:.2f}% +/- {100 * s['test_population_std']:.2f}%; 10 fits; {status['elapsed_seconds']:.1f} s")
    print("Verified matching release, inputs, dependencies, configuration and both 10-fit population summaries.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one immutable-release demo and retain its complete provenance."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", choices=["cpu", "gpu"])
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    source = base / "code"
    folder = base / "runs" / args.device
    folder.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert commit == "ecbbd608a4c915359d8c99b38fe875eb614aa8ce"
    assert not (folder / "demo-summary.json").exists(), "Do not overwrite a completed run"
    files = sorted([*source.glob("graphslim/**/*.py"), *source.glob("graphslim/configs/**/*.json"), source / "examples/demo.py"])
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    data = Path("/home/sgong36/gslim/data/cora")
    data_hashes = {str(p.relative_to(data)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.rglob("*")) if p.is_file()}
    lock = None
    if args.device == "gpu":
        assert args.gpu_uuid
        lock = open(f"/tmp/graphslim-{args.gpu_uuid}.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = subprocess.check_output(["nvidia-smi", f"--id={args.gpu_uuid}", "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv"], text=True)
        (folder / "nvidia_before.csv").write_text(before)
    import torch
    torch.set_num_threads(4)
    if args.device == "gpu":
        assert torch.cuda.is_available()
        check = torch.eye(3, device="cuda:0")
        assert torch.equal(check @ check, check)
        torch.cuda.synchronize()
    environment = {
        "source_version": "1.3.4",
        "source_commit": commit,
        "source_sha256": hashes,
        "input_files_sha256": data_hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "packages": {name: importlib.metadata.version(name) for name in ["graphslim", "torch", "torch-geometric", "torch-sparse", "torch-scatter", "numpy", "scipy", "scikit-learn", "dgl", "networkit"]},
        "device": args.device,
        "physical_gpu_uuid": args.gpu_uuid,
        "torch_cuda_runtime": torch.version.cuda,
        "hardware": torch.cuda.get_device_name(0) if args.device == "gpu" else platform.processor(),
        "cpu_model": next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), None),
        "environment_options": {name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED"]},
        "scope": "One seed-1 GCond reduction with complete defaults, followed by ten validation-selected GAT fits. Integration example; not an all-method GPU test or an isolated hardware speed benchmark.",
    }
    assert environment["packages"]["graphslim"] == "1.3.4"
    command = [sys.executable, "examples/demo.py", "-G", "0" if args.device == "gpu" else "-1", "--load_path", "/home/sgong36/gslim/data", "--save_path", str(folder)]
    environment["command"] = command
    environment["started_utc"] = datetime.now(timezone.utc).isoformat()
    write(folder / "provenance.json", environment)
    write(folder / "status.json", {"state": "running", "pid": os.getpid(), "started_utc": environment["started_utc"]})
    tick = time.monotonic()
    with (folder / "console.txt").open("w") as output:
        process = subprocess.run(command, cwd=source, stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy())
    status = {"state": "complete" if process.returncode == 0 else "failed", "exit_code": process.returncode, "elapsed_seconds": time.monotonic() - tick, "finished_utc": datetime.now(timezone.utc).isoformat()}
    if process.returncode == 0:
        summary = json.loads((folder / "demo-summary.json").read_text())
        scores = [r["test"] for r in summary["fits"]]
        assert len(scores) == 10
        assert summary["config"]["epochs"] == 1000
        assert summary["config"]["outer_loop"] == 20 and summary["config"]["inner_loop"] == 15
        assert summary["config"]["eval_epochs"] == 300
        assert abs(statistics.mean(scores) - summary["test_mean"]) < 1e-12
        assert abs(statistics.pstdev(scores) - summary["test_population_std"]) < 1e-12
        assert all(hashlib.sha256((source / n).read_bytes()).hexdigest() == h for n, h in hashes.items())
        status.update(test_mean=summary["test_mean"], test_population_std=summary["test_population_std"], verified_fits=10, source_unchanged=True)
    write(folder / "status.json", status)
    print(json.dumps(status), flush=True)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())

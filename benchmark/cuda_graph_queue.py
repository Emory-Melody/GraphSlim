"""Freeze CUDA graph validation, then wait for genuinely idle, unlocked GPUs.

Never interrupts another queue. Results are integration evidence, not scientific
reproduction tables. Each job has a terminal record and an isolated source root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from benchmark.rerun_nr import (
    acquire_gpu,
    gpu_inventory,
    lock_file,
    source_files,
    source_digest,
    write_json,
)
from graphslim.reduction import list_methods

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--poll-seconds", type=int, default=15)
    options = parser.parse_args()
    folder = options.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / "frozen"
    manifest_path = folder / "manifest.json"
    # Preserve the virtualenv path: resolving its interpreter symlink loses the
    # adjacent pyvenv.cfg and launches a Python without the installed Torch/PyG.
    python = Path(sys.executable).absolute()
    with lock_file(folder / "manager.lock", blocking=False):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            files = sorted(
                set(
                    source_files()
                    + [
                        ROOT / "benchmark/__init__.py",
                        ROOT / "benchmark/verify_cuda_graph.py",
                        ROOT / "benchmark/cuda_graph_queue.py",
                        ROOT / "benchmark/verify_lightweight.py",
                        ROOT / "tests/test_acceleration.py",
                    ]
                )
            )
            for name in ("sgnk.py", "gpr.py"):
                reference = ROOT / "runs/reference/GCGP/gcgp" / name
                if reference.exists():
                    files.append(reference)
            with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    archive.write(path, path.relative_to(ROOT))
            with zipfile.ZipFile(folder / "source.zip") as archive:
                archive.extractall(frozen)
            jobs = [
                {
                    "id": "unit_gpu",
                    "argv": [
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_acceleration.py",
                        "--junitxml=" + str(folder / "unit_gpu.xml"),
                    ],
                }
            ]
            methods = list(list_methods())
            for index in range(3):
                roster = methods[index::3]
                jobs.append(
                    {
                        "id": f"all_methods_gpu_{index}",
                        "argv": [
                            "-m",
                            "benchmark.verify_cuda_graph",
                            "--steps",
                            "80",
                            "--methods",
                            *roster,
                            "--output",
                            str(folder / f"all_methods_gpu_{index}.json"),
                        ],
                    }
                )
            for setting in ("trans", "ind"):
                jobs.append(
                    {
                        "id": f"noise_{setting}",
                        "argv": [
                            "-m",
                            "benchmark.verify_cuda_graph",
                            "--steps",
                            "80",
                            "--methods",
                            "cgc",
                            "clustgdd",
                            "gcgp",
                            "nope",
                            "--setting",
                            setting,
                            "--noise",
                            "random_adj",
                            "edge_drop",
                            "random_feat",
                            "feature_mask",
                            "gaussian",
                            "label_noise",
                            "--output",
                            str(folder / f"noise_{setting}.json"),
                        ],
                    }
                )
            manifest = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": source_digest(files),
                "source_archive_sha256": hashlib.sha256(
                    (folder / "source.zip").read_bytes()
                ).hexdigest(),
                "scope": "Native CUDA graph integration: 40 method tags x 7 downstream models; 4 lightweight tags x 6 corruptions x 7 models in trans/ind. Synthetic inputs, not actual reducer reproductions or benchmark accuracy.",
                "jobs": jobs,
                "python": str(python),
                "gpu_indices": options.gpus,
            }
            write_json(manifest_path, manifest)
        active = {}
        while True:
            for job_id, (process, handles, log, gpu) in list(active.items()):
                code = process.poll()
                if code is not None:
                    write_json(
                        folder / f"{job_id}.terminal.json",
                        {
                            "job": job_id,
                            "status": "complete" if code == 0 else "failed",
                            "exit_code": code,
                            "gpu": gpu,
                            "source_sha256": manifest["source_sha256"],
                        },
                    )
                    log.close()
                    for handle in handles:
                        handle.close()
                    del active[job_id]
            pending = [
                job
                for job in manifest["jobs"]
                if not (folder / f"{job['id']}.terminal.json").exists() and job["id"] not in active
            ]
            # The GPU unit suite must pass before the broad model matrix starts.
            unit = folder / "unit_gpu.terminal.json"
            if unit.exists() and json.loads(unit.read_text())["status"] != "complete":
                write_json(
                    folder / "state.json", {"status": "unit_tests_failed", "active": list(active)}
                )
                raise SystemExit(1)
            if not unit.exists():
                pending = [job for job in pending if job["id"] == "unit_gpu"]
            for gpu in gpu_inventory():
                if not pending or not gpu["idle"] or gpu["index"] not in options.gpus:
                    continue
                handles = acquire_gpu(gpu)
                if handles is None:
                    continue
                job = pending.pop(0)
                log = (folder / f"{job['id']}.log").open("a")
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu["uuid"],
                    PYTHONPATH=str(frozen),
                    PYTHONUNBUFFERED="1",
                )
                process = subprocess.Popen(
                    [str(python), *job["argv"]],
                    cwd=frozen,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=tuple(h.fileno() for h in handles),
                )
                active[job["id"]] = process, handles, log, gpu
            terminals = [json.loads(path.read_text()) for path in folder.glob("*.terminal.json")]
            status = "running" if active else "waiting_for_idle_gpu"
            if len(terminals) == len(manifest["jobs"]):
                status = (
                    "complete" if all(t["status"] == "complete" for t in terminals) else "failed"
                )
            write_json(
                folder / "state.json",
                {
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "complete": sum(t["status"] == "complete" for t in terminals),
                    "failed": sum(t["status"] == "failed" for t in terminals),
                    "expected_jobs": len(manifest["jobs"]),
                    "active": list(active),
                    "source_sha256": manifest["source_sha256"],
                },
            )
            if status in ("complete", "failed"):
                if status == "failed":
                    raise SystemExit(1)
                break
            time.sleep(options.poll_seconds)


if __name__ == "__main__":
    main()

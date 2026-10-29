"""Continue evaluation-only timeouts using their original frozen computation.

Creates a separate run, retaining all original terminal records. Only wall/RSS
limits may change; completed fits and reductions are hash-verified and copied.
"""

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

from benchmark import expand_datasets as expansion
from benchmark import rerun_nr as nr


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(parent, manifest):
    if digest(parent / "source.zip") != manifest["archive_sha256"]:
        raise ValueError("Parent source archive hash mismatch")
    source = hashlib.sha256()
    with zipfile.ZipFile(parent / "source.zip") as archive:
        for name in sorted(archive.namelist()):
            content = archive.read(name)
            source.update(name.encode())
            source.update(content)
            if (parent / "frozen_workspace" / name).read_bytes() != content:
                raise ValueError(f"Frozen workspace differs from archive: {name}")
    if source.hexdigest() != manifest["source_sha256"]:
        raise ValueError("Parent source digest mismatch")


def saved_files(folder, job, manifest):
    """Validate all reusable science inputs before creating a continuation."""
    source = expansion.job_source(manifest, job)
    if source != manifest["source_sha256"]:
        raise ValueError("Cannot resume a job from a different inherited source")
    data = nr.read_json(folder / "dataset.json")
    files = {"dataset.json": digest(folder / "dataset.json")}

    def check(record):
        if record["source_sha256"] != source or record["job"] != job or record["data"] != data:
            raise ValueError("Saved source/data/job mismatch")

    checkpoints = None
    if job["kind"] == "reduction":
        saved = nr.read_json(folder / "reduction.json")
        check(saved)
        files["reduction.json"] = digest(folder / "reduction.json")
        checkpoints = saved["checkpoint_sha256"]
        if not checkpoints:
            raise ValueError("Missing checkpoints")
        for name, expected in checkpoints.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid checkpoint path")
            if digest(folder / name) != expected:
                raise ValueError("Saved checkpoint hash mismatch")
            files[name] = expected
    seeds = []
    for path in sorted((folder / "evaluations").glob("*.json")):
        saved = nr.read_json(path)
        check(saved)
        seed = saved["score"]["seed"]
        if (
            seed not in manifest["protocol"]["evaluation_seeds"]
            or path.name != f"GCN__s{seed}.json"
        ):
            raise ValueError("Saved evaluation seed mismatch")
        if saved.get("checkpoint_sha256") != checkpoints:
            raise ValueError("Saved evaluation checkpoint mismatch")
        if not all(
            math.isfinite(saved["score"][key]) and 0 <= saved["score"][key] <= 1
            for key in ("validation", "test")
        ):
            raise ValueError("Invalid saved score")
        seeds.append(seed)
        files[str(path.relative_to(folder))] = digest(path)
    if not seeds:
        raise ValueError("No completed evaluations to resume")
    return files, seeds


def initialize(parent, folder, timeout, max_rss_gib, gpus):
    parent, folder = parent.resolve(), folder.resolve()
    original = nr.read_json(parent / "manifest.json")
    verify_snapshot(parent, original)
    if folder.exists():
        current = nr.read_json(folder / "manifest.json")
        continuation = current["continuation"]
        if (
            continuation["parent"] != str(parent)
            or continuation["parent_manifest_sha256"] != digest(parent / "manifest.json")
            or current["protocol"]["timeout_seconds"] != timeout
            or current["protocol"]["max_rss_gib"] != max_rss_gib
            or current["gpus"] != gpus
        ):
            raise ValueError("Continuation configuration changed; use a fresh directory")
        for job_id, audit in continuation["jobs"].items():
            for name, expected in audit["copied_sha256"].items():
                if digest(folder / "jobs" / job_id / name) != expected:
                    raise ValueError("Continuation saved input changed")
        return current
    retries = {}
    for job in original["jobs"]:
        terminal = expansion.terminal(parent, job)
        if terminal is None or expansion.worker_alive(parent, job):
            raise ValueError("Parent must be terminal with no surviving worker")
        if terminal["job"] != job or terminal["source_sha256"] != expansion.job_source(
            original, job
        ):
            raise ValueError("Parent terminal source/job mismatch")
        path = parent / "jobs" / job["id"]
        heartbeat = (
            nr.read_json(path / "heartbeat.json") if (path / "heartbeat.json").exists() else {}
        )
        if (
            terminal["status"] != "timeout"
            or job["kind"] == "parity"
            or not heartbeat.get("phase", "").startswith("evaluation_")
        ):
            continue
        files, seeds = saved_files(path, job, original)
        retries[job["id"]] = dict(parent_terminal=terminal, copied_sha256=files, reused_seeds=seeds)
    if not retries:
        raise ValueError("No evaluation-only timeouts found")
    current = deepcopy(original)
    current["created_utc"] = nr.utcnow()
    current["protocol"].update(timeout_seconds=timeout, max_rss_gib=max_rss_gib)
    current["gpus"] = gpus
    current["continuation"] = dict(
        parent=str(parent),
        parent_manifest_sha256=digest(parent / "manifest.json"),
        parent_limits={
            key: original["protocol"][key] for key in ("timeout_seconds", "max_rss_gib")
        },
        jobs=retries,
        supervisor_sha256=digest(Path(__file__)),
        policy="Same frozen code/data/seeds/checkpoints; only execution limits changed. Original terminal records retained; only missing evaluations run.",
    )
    folder.mkdir(parents=True)
    for name in ("source.zip", "frozen_workspace"):
        (folder / name).symlink_to(parent / name)
    (folder / "jobs").mkdir()
    for job in original["jobs"]:
        source = parent / "jobs" / job["id"]
        target = folder / "jobs" / job["id"]
        if job["id"] not in retries:
            target.symlink_to(source, target_is_directory=True)
            continue
        target.mkdir()
        for name, expected in retries[job["id"]]["copied_sha256"].items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
            if digest(destination) != expected:
                raise ValueError("Input changed while copying")
    nr.write_json(folder / "manifest.json", current)
    expansion.export(folder, current)
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--max-rss-gib", type=float, default=100)
    parser.add_argument("--gpus", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.max_rss_gib <= 0:
        parser.error("Resource limits must be positive")
    manifest = initialize(args.parent, args.run_dir, args.timeout, args.max_rss_gib, args.gpus)
    if not args.initialize_only:
        expansion.manage(
            SimpleNamespace(
                run_dir=args.run_dir.resolve(),
                timeout=args.timeout,
                max_rss_gib=args.max_rss_gib,
                gpus=args.gpus,
                threads=manifest["protocol"]["threads"],
            ),
            manifest,
        )


if __name__ == "__main__":
    main()

"""A continuation must preserve inputs, original failures and complete-only scores."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from benchmark import resume_expansion as resume


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parent_run(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    name, content = "graphslim/example.py", b"# frozen\n"
    with zipfile.ZipFile(parent / "source.zip", "w") as archive:
        archive.writestr(name, content)
    path = parent / "frozen_workspace" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    source = hashlib.sha256(name.encode() + content).hexdigest()
    jobs = [
        dict(
            id="whole__cora",
            stage=1,
            dataset="cora",
            method="whole",
            kind="whole",
            seed=1,
            rate=None,
        ),
        dict(
            id="cora__nope__r0.1__s1",
            stage=1,
            dataset="cora",
            method="nope",
            kind="reduction",
            seed=1,
            rate="0.1",
        ),
    ]
    manifest = dict(
        source_sha256=source,
        archive_sha256=resume.digest(parent / "source.zip"),
        jobs=jobs,
        protocol=dict(
            evaluation_seeds=[0, 1],
            threads=4,
            timeout_seconds=3600,
            max_rss_gib=100,
            eval_epochs=300,
        ),
    )
    write(parent / "manifest.json", manifest)
    for job in jobs:
        folder = parent / "jobs" / job["id"]
        write(folder / "terminal.json", dict(job=job, source_sha256=source, status="timeout"))
        write(
            folder / "heartbeat.json",
            dict(pid=-1, phase="evaluation_1" if job["kind"] == "whole" else "reduction"),
        )
    folder = parent / "jobs" / jobs[0]["id"]
    write(folder / "dataset.json", {"nodes": 3})
    write(
        folder / "evaluations" / "GCN__s0.json",
        dict(
            source_sha256=source,
            job=jobs[0],
            data={"nodes": 3},
            checkpoint_sha256=None,
            score=dict(seed=0, validation=0.5, test=0.6, seconds=2),
            resources={},
        ),
    )
    return parent, manifest


def test_resume_copies_only_verified_work_and_keeps_original_terminal(tmp_path):
    parent, original = parent_run(tmp_path)
    output = tmp_path / "resume"
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    manifest = resume.initialize(parent, output, 14400, 100, [3])
    assert manifest["source_sha256"] == original["source_sha256"]
    assert manifest["protocol"] == {**original["protocol"], "timeout_seconds": 14400}
    assert manifest["continuation"]["jobs"]["whole__cora"]["reused_seeds"] == [0]
    assert not (output / "jobs/whole__cora/terminal.json").exists()
    assert not (output / "jobs/whole__cora").is_symlink()
    assert (output / "jobs/cora__nope__r0.1__s1").is_symlink()
    assert json.loads((output / "summary.json").read_text())["numeric_cells"] == 0
    assert resume.initialize(parent, output, 14400, 100, [3]) == manifest
    assert before == {
        str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("corruption", ["source", "dataset", "seed", "checkpoint"])
def test_resume_rejects_corrupt_inputs_before_creating_run(tmp_path, corruption):
    parent, _ = parent_run(tmp_path)
    if corruption == "source":
        (parent / "frozen_workspace/graphslim/example.py").write_text("changed")
    else:
        path = parent / "jobs/whole__cora/evaluations/GCN__s0.json"
        value = json.loads(path.read_text())
        if corruption == "dataset":
            value["data"]["nodes"] = 7
        elif corruption == "seed":
            value["score"]["seed"] = 9
        else:
            value["checkpoint_sha256"] = {"unrelated": "hash"}
        write(path, value)
    output = tmp_path / "resume"
    with pytest.raises(ValueError):
        resume.initialize(parent, output, 14400, 100, [3])
    assert not output.exists()


def test_unavailable_gpu_does_not_block_valid_devices(monkeypatch):
    def query(command, **kwargs):
        if "--query-compute-apps=gpu_uuid" in command:
            return "GPU-busy\n"
        return "0, GPU-off, 0, [N/A]\n1, GPU-busy, 0, 0\n3, GPU-free, 0, 0\n"

    monkeypatch.setattr(resume.nr.subprocess, "check_output", query)
    assert [gpu["idle"] for gpu in resume.nr.gpu_inventory()] == [False, False, True]

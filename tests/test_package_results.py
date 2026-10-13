import hashlib
import math
import zipfile

import pytest

from benchmark import package_results as package
from benchmark import paper_hpo as hpo
from benchmark import rerun_nr as nr


def recorded_cell(tmp_path):
    jobs = nr.plan_jobs(["cora"], ["clustering"])[:3]
    manifest = {"source_sha256": "frozen", "jobs": jobs}
    entry = {"format": "nr", "path": "runs/example"}
    values = []
    for job in jobs:
        evaluations = [
            {"seed": seed, "validation": 0.6, "test": 0.5 + seed * 0.01}
            for seed in range(job["seed"], job["seed"] + 10)
        ]
        values.extend(item["test"] for item in evaluations)
        nr.write_json(
            nr.job_path(tmp_path, job) / "result.json",
            {
                "job": job,
                "source_sha256": "frozen",
                "data": {"split": "fixed"},
                "evaluations": evaluations,
            },
        )
    return manifest, entry, values


def test_package_recomputes_from_all_raw_scores(tmp_path):
    manifest, entry, values = recorded_cell(tmp_path)
    cell = package.collect_cell(tmp_path, manifest, entry, "clustering", "cora", "0.1")
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    assert cell["n"] == 30 and cell["status"] == "complete"
    assert cell["mean"] == pytest.approx(mean)
    assert cell["std"] == pytest.approx(spread)
    assert len(cell["result_sha256"]) == 3


@pytest.mark.parametrize("corruption", ["source", "split", "duplicate_seed", "out_of_range"])
def test_package_rejects_corrupted_complete_cells(tmp_path, corruption):
    manifest, entry, _ = recorded_cell(tmp_path)
    path = nr.job_path(tmp_path, manifest["jobs"][0]) / "result.json"
    record = nr.read_json(path)
    if corruption == "source":
        record["source_sha256"] = "other"
    elif corruption == "split":
        record["data"]["split"] = "other"
    elif corruption == "duplicate_seed":
        record["evaluations"][0]["seed"] = record["evaluations"][1]["seed"]
    else:
        record["evaluations"][0]["test"] = 1.5
    nr.write_json(path, record)
    with pytest.raises(RuntimeError):
        package.collect_cell(tmp_path, manifest, entry, "clustering", "cora", "0.1")


def test_pending_cell_has_no_numeric_estimate(tmp_path):
    jobs = nr.plan_jobs(["cora"], ["clustering"])[:3]
    cell = package.collect_cell(
        tmp_path,
        {"source_sha256": "frozen", "jobs": jobs},
        {"format": "nr", "path": "runs/new"},
        "clustering",
        "cora",
        "0.1",
    )
    assert cell["n"] == 0 and "mean" not in cell and "std" not in cell
    assert cell["value"] == "PENDING (0/3)"


def test_archive_verification_uses_frozen_bytes_not_live_sources(tmp_path):
    digest = hashlib.sha256()
    with zipfile.ZipFile(tmp_path / "source.zip", "w") as archive:
        for name, content in (
            ("graphslim/model.py", b"frozen\n"),
            ("benchmark/runner.py", b"runner\n"),
        ):
            archive.writestr(name, content)
            digest.update(name.encode())
            digest.update(content)
    assert len(package.verify_archive(tmp_path, digest.hexdigest())) == 64
    with pytest.raises(RuntimeError, match="archive"):
        package.verify_archive(tmp_path, "different")


def test_new_pending_assignment_never_falls_back_to_superseded_run(tmp_path):
    entries = [
        {
            "path": "old",
            "format": "nr",
            "methods": list(nr.METHODS),
            "datasets": list(package.RATES),
            "label": "old",
        },
        {
            "path": "new",
            "format": "hpo",
            "methods": ["simgc"],
            "datasets": list(package.RATES),
            "label": "new",
        },
    ]
    catalog = tmp_path / "catalog.json"
    nr.write_json(catalog, {"runs": entries})
    cells, _ = package.collect(catalog, root=tmp_path)
    assert len(cells) == 105
    simgc = [cell for cell in cells if cell["method"] == "simgc"]
    assert len(simgc) == 21 and all(cell["run"] == "new" for cell in simgc)


def test_hpo_scoped_roster_keeps_all_dependencies_and_excludes_gecc():
    jobs = hpo.roster(methods=["simgc"])
    assert len(jobs) == 191 and {job["method"] for job in jobs} == {"simgc"}
    assert sum(len(job["candidates"]) for job in jobs if job["kind"] == "gcn") == 758
    assert all(
        dependency in {job["id"] for job in jobs} for job in jobs for dependency in job["depends"]
    )
    citation = hpo.roster(["cora", "citeseer"], ["simgc"])
    assert len(citation) == 56
    with pytest.raises(ValueError):
        hpo.roster(["cora"], ["gecc"])


def test_scoped_hpo_export_does_not_treat_absent_cells_as_complete(tmp_path):
    manifest = hpo.initialize(tmp_path, ["cora"], ["simgc"], "repaired-hpo")
    states, _ = hpo.export(tmp_path, manifest)
    assert len(states) == 28
    cells = nr.read_json(tmp_path / "summary.json")["cells"]
    assert len(cells) == 3 and all(cell["value"] == "PENDING" for cell in cells)
    with pytest.raises(RuntimeError, match="scope"):
        hpo.initialize(tmp_path, ["citeseer"], ["simgc"])

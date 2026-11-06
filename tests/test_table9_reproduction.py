import json

from benchmark import render_reproduction as render


def test_historical_rows_keep_provenance_without_restoring_removed_method():
    cells = [
        dict(
            dataset="cora",
            method="gcondx",
            rate=rate,
            status="complete",
            mean=80.0,
            std=1.0,
            paper=80.0,
        )
        for rate in render.RATES["cora"]
    ]
    snapshot = dict(cells=cells, replay={"gcondx": {"method": "doscondx", "config": "gcondx"}})
    before = json.dumps(snapshot)
    indexed = render.index(snapshot)
    assert {key[1] for key in indexed} == {"doscondx"}
    assert all(c["legacy_config"] == "gcondx" for c in indexed.values())
    assert "| Condensation | DosCondX¹ |" in render.readme_section(snapshot)
    assert "| Condensation | GCondX" not in render.readme_section(snapshot)
    assert json.dumps(snapshot) == before
    # The removed native implementation must never be relabelled as DosCondX.
    assert render.index(dict(cells=cells, replay={})) == {}


def fake_run(folder, method, scores_by_rate, dataset="cora"):
    jobs = [
        dict(
            id=f"{dataset}__{method}__r{rate}__s1",
            dataset=dataset,
            method=method,
            rate=rate,
            seed=1,
        )
        for rate in scores_by_rate
    ]
    manifest = dict(source_sha256="abc", protocol={}, replay={}, jobs=jobs)
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    for job in jobs:
        path = folder / "jobs" / job["id"]
        path.mkdir(parents=True)
        score = scores_by_rate[job["rate"]]
        record = dict(
            elapsed_seconds=1.0,
            reduced_nodes=7,
            evaluations=[{"seed": i, "test": score / 100} for i in range(10)],
        )
        (path / "result.json").write_text(json.dumps(record))


def test_original_row_selection_omits_distant_rows_and_failure_commentary(tmp_path, monkeypatch):
    # Cora GCond in Table 9: 80.17, 80.81, 80.54. DosCond: 80.65, 80.85, 81.15.
    fake_run(tmp_path / "close", "gcond", {"0.1": 80.0, "0.25": 81.0, "0.5": 80.0})
    fake_run(tmp_path / "far", "doscond", {"0.1": 70.0, "0.25": 71.0, "0.5": 72.0})
    monkeypatch.setattr(render, "ROOT", tmp_path)
    monkeypatch.setattr(render, "SNAPSHOT", tmp_path / "snapshot.json")
    monkeypatch.setattr(
        render, "SOURCES", [dict(run="close", methods=None), dict(run="far", methods=None)]
    )
    snapshot = render.collect()
    cells = render.index(snapshot)
    assert render.row_status(cells, "cora", "gcond")[0] == "reproduced"
    assert render.row_status(cells, "cora", "doscond")[0] == "outside limit"
    section = render.readme_section(snapshot, "en")
    cora = section.split("#### Cora —", 1)[1].split("#### Pubmed —", 1)[0]
    assert (
        "| Condensation | GCond | Archived run / SD | 80.00 ± 0.00 | 81.00 ± 0.00 | 80.00 ± 0.00 |"
        in section
    )
    assert "| Condensation | DosCond |" not in cora
    assert "at most 3 percentage points" in section
    assert "Rows more than" not in section and "Runs that failed" not in section
    assert "| 凝聚 | GCond |" in render.readme_section(snapshot, "zh")
    for cell in snapshot["cells"]:
        if cell["method"] == "gcond":
            cell.update(status="failed", kind="OOM")
    section = render.readme_section(snapshot, "en")
    cora = section.split("#### Cora —", 1)[1].split("#### Pubmed —", 1)[0]
    assert "| Condensation | GCond |" not in cora
    assert "OOM" not in section and "failed" not in section.lower()


def test_readmes_match_the_frozen_snapshot():
    snapshot = json.loads(render.SNAPSHOT.read_text(encoding="utf-8"))
    for path, make in render.targets(snapshot).items():
        assert path.read_text(encoding="utf-8") == make(), path.name


def test_yelp_ap_never_substitutes_historical_f1_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ROOT", tmp_path)
    old = dict(
        cells=[
            dict(
                dataset="yelp",
                method="clustering",
                rate=rate,
                status="complete",
                mean=53.63,
                std=0.36,
                paper=None,
            )
            for rate in render.RATES["yelp"]
        ]
    )
    section, _ = render.readme_tables(old)
    assert "#### Yelp — AP (%)" in section
    assert "| Coarsening | Clustering | NR | NR | NR |" in section
    assert "53.63" not in section


def test_yelp_ap_rejects_wrong_metric_and_withholds_partial_rows():
    import pytest

    cells = [
        dict(
            method="clustering",
            rate=rate,
            status="complete",
            metric="F1-macro",
            mean=53.63,
            std=0.36,
        )
        for rate in render.RATES["yelp"]
    ]
    with pytest.raises(ValueError, match="Non-AP measurement"):
        render.readme_yelp_ap(dict(cells=cells))
    for cell in cells:
        cell["metric"] = "AP"
    cells[-1]["status"] = "incomplete"
    assert "| Coarsening | Clustering | NR | NR | NR |" in render.readme_yelp_ap(dict(cells=cells))

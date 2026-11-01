"""Resumable validation-only grid search over reduction settings."""

import json

import pytest

from graphslim import hpo
from test_lightweight_methods import quick_args, tiny_graph


def search_args(tmp_path, method="kcenter"):
    args = quick_args(tmp_path, method, "trans")
    args.hpo, args.hpo_yes, args.hpo_eval_runs, args.hpo_space = True, True, 2, None
    args.hidden = 8
    return args


def test_template_is_written_and_a_single_point_does_not_run(tmp_path):
    args = search_args(tmp_path)
    assert hpo.run_search(args, tiny_graph()) is None
    folder = hpo.search_dir(args)
    template = json.loads((folder / "space.json").read_text())
    assert "_comment" in template and all(
        isinstance(v, list) for k, v in template.items() if not k.startswith("_")
    )
    assert not (folder / "trials.jsonl").exists()


def test_grid_runs_records_validation_only_and_resumes(tmp_path):
    args = search_args(tmp_path)
    space = tmp_path / "space.json"
    space.write_text(json.dumps({"hidden": [8, 16], "_comment": "x"}))
    args.hpo_space = str(space)
    best = hpo.run_search(args, tiny_graph())
    folder = hpo.search_dir(args)
    lines = [json.loads(l) for l in (folder / "trials.jsonl").read_text().splitlines()]
    assert [r["trial"] for r in lines] == [0, 1] and all(r["status"] == "complete" for r in lines)
    assert all(len(r["validation"]) == 2 and not {k for k in r if "test" in k} for r in lines)
    assert best["params"] in ({"hidden": 8}, {"hidden": 16})
    assert json.loads((folder / "best.json").read_text()) == best["params"]
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["trials"] == 2 and summary["complete"] == 2
    # Resume: nothing new runs, the record file is unchanged, the same winner is reported.
    again = hpo.run_search(_resume_args(tmp_path, space), tiny_graph())
    assert again["trial"] == best["trial"]
    assert (folder / "trials.jsonl").read_text().count("\n") == 2
    assert json.loads((folder / "summary.json").read_text())["trials_run_this_session"] == 0


def _resume_args(tmp_path, space):
    args = search_args(tmp_path)
    args.hpo_space = str(space)
    return args


def test_forbidden_and_malformed_spaces_are_rejected(tmp_path):
    import click

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"seed": [1, 2]}))
    with pytest.raises(click.BadParameter):
        hpo.load_space(bad)
    bad.write_text(json.dumps({"hidden": 8}))
    with pytest.raises(click.BadParameter):
        hpo.load_space(bad)


def test_failed_trials_are_recorded_and_skipped_for_selection(tmp_path, monkeypatch):
    args = search_args(tmp_path)
    space = tmp_path / "space.json"
    space.write_text(json.dumps({"hidden": [8, 16]}))
    args.hpo_space = str(space)
    import graphslim.hpo as module

    original = module.validation_score
    calls = []

    def flaky(graph, run_args, runs):
        calls.append(run_args.hidden)
        if run_args.hidden == 8:
            raise RuntimeError("boom")
        return original(graph, run_args, runs)

    monkeypatch.setattr(module, "validation_score", flaky)
    best = hpo.run_search(args, tiny_graph())
    lines = [
        json.loads(l) for l in (hpo.search_dir(args) / "trials.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in lines] == ["failed", "complete"] and "boom" in lines[0]["error"]
    assert best["params"] == {"hidden": 16}

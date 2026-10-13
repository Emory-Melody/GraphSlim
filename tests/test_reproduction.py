import pytest

from benchmark.render_results import read_results, render_results, ROOT, START, END
from benchmark.reproduce import plan, main, PERFORMANCE_METHODS, RATES, SUITES


@pytest.mark.parametrize("method", ("cgc", "clustgdd", "gcgp", "nope"))
@pytest.mark.parametrize("suite", SUITES)
def test_new_methods_are_respected_by_every_suite(method, suite):
    dataset = "ogbn-arxiv" if suite == "scalability" else "cora"
    jobs = plan(suite, datasets=[dataset], methods=[method], whole=False)
    assert jobs
    for job in jobs:
        arguments = list(job.arguments)
        assert arguments[arguments.index("-M") + 1] == method
        assert arguments[arguments.index("-D") + 1] == dataset


def test_performance_covers_all_datasets_and_methods_without_duplicates():
    jobs = plan("performance")
    assert len(jobs) == len(RATES) * (3 * len(PERFORMANCE_METHODS) + 1)
    assert len(jobs) == len(set(jobs))
    assert any("-D" in job.arguments and "citeseer" in job.arguments for job in jobs)
    assert any("clustering" in job.arguments for job in jobs)


def test_dry_run_never_starts_training(monkeypatch, capsys):
    import benchmark.reproduce as reproduce

    def fail(*args, **kwargs):
        raise AssertionError("dry run launched a subprocess")

    monkeypatch.setattr(reproduce.subprocess, "run", fail)
    main(
        [
            "performance",
            "--datasets",
            "cora",
            "--methods",
            "clustering",
            "--dry-run",
            "--",
            "-G",
            "-1",
        ]
    )
    output = capsys.readouterr().out
    assert "4 jobs planned." in output
    assert "-G -1" in output


def test_slurm_dry_run_does_not_submit(monkeypatch, capsys):
    import benchmark.reproduce as reproduce

    def fail(*args, **kwargs):
        raise AssertionError("dry run submitted a job")

    monkeypatch.setattr(reproduce.subprocess, "run", fail)
    main(
        [
            "performance",
            "--datasets",
            "cora",
            "--methods",
            "clustering",
            "--slurm",
            "--dry-run",
            "--slurm-arg=--gpus=1",
        ]
    )
    assert "sbatch --gpus=1 --wrap" in capsys.readouterr().out


def test_published_results_preserve_values_missingness_and_readme():
    rows = read_results()
    assert len({row["Dataset"] for row in rows}) == 7
    cora = next(
        row for row in rows if row["Dataset"] == "Cora" and row["Reduction rate (%)"] == "0.50"
    )
    assert cora["GCond"] == "80.17±0.8"
    assert cora["Clustering"] == cora["GCDM"] == "NR"
    reddit = next(row for row in rows if row["Dataset"] == "Reddit")
    assert reddit["GCSNTK"] == "OOM"
    assert reddit["GDEM"] == "-"
    readme = (ROOT / "README.md").read_text()
    assert readme.split(START)[1].split(END)[0].strip() == render_results(rows).strip()

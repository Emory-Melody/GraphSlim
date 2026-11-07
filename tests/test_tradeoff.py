# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import json
from pathlib import Path

import pytest

from benchmark.render_tradeoff import (
    DATASETS,
    METHODS,
    SNAPSHOT,
    collect,
    markdown_table,
    render,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {"reduction_seeds": [1, 2, 3], "evaluation_seeds": list(range(10))}


def fake_run(root):
    """Write the smallest run directory that ``collect`` accepts."""
    (root / "manifest.json").write_text(
        json.dumps({"protocol": PROTOCOL, "source_sha256": "source", "archive_sha256": "archive"})
    )

    def job(job_id, method, seconds):
        folder = root / "jobs" / job_id
        folder.mkdir(parents=True)
        record = {
            "job": {"id": job_id},
            "source_sha256": "source",
            "gpu_name": "GPU",
            "runtime": {"torch": "2.8.0", "cuda": "12.8"},
            "data": {"nodes": 1000},
            "reduction": {"synthetic_nodes": 10},
            "evaluations": [
                {"seed": seed, "test": 0.5 + seed / 100} for seed in PROTOCOL["evaluation_seeds"]
            ],
            "timing": {"reduction_seconds": 0.0 if method == "whole" else seconds},
        }
        (folder / "result.json").write_text(json.dumps(record))

    for dataset, rates in DATASETS.items():
        job(f"whole__{dataset}", "whole", 0.0)
        for method in METHODS:
            for rate in rates:
                for seed in PROTOCOL["reduction_seeds"]:
                    job(f"{dataset}__{method}__r{rate}__s{seed}", method, float(seed))


def test_collect_summarizes_complete_cells(tmp_path):
    fake_run(tmp_path)
    snapshot = collect(tmp_path)
    reduced = [cell for cell in snapshot["cells"] if cell["method"] != "whole"]
    assert len(snapshot["cells"]) == len(DATASETS) * (1 + len(METHODS) * 3)
    assert {cell["n_evaluations"] for cell in reduced} == {30}
    assert {cell["median_reduction_seconds"] for cell in reduced} == {2.0}
    assert snapshot["hardware"] == {"gpu": "GPU", "torch": "2.8.0", "cuda": "12.8"}


def test_collect_refuses_missing_evaluation_seed(tmp_path):
    fake_run(tmp_path)
    path = tmp_path / "jobs" / "reddit__cgc__r0.001__s2" / "result.json"
    record = json.loads(path.read_text())
    record["evaluations"].pop()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="every evaluation seed"):
        collect(tmp_path)


def test_collect_refuses_mixed_hardware(tmp_path):
    fake_run(tmp_path)
    path = tmp_path / "jobs" / "whole__reddit" / "result.json"
    record = json.loads(path.read_text())
    record["gpu_name"] = "Other GPU"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mix hardware"):
        collect(tmp_path)


def test_frozen_snapshot_is_complete():
    snapshot = json.loads(SNAPSHOT.read_text())
    for cell in snapshot["cells"]:
        expected = 10 if cell["method"] == "whole" else 30
        assert cell["n_evaluations"] == expected
        assert len(cell["result_sha256"]) == len(cell["job_ids"])


@pytest.mark.parametrize(
    "language, header, whole_label",
    [
        (
            "en",
            "| Dataset | Method | Budget | Reduced nodes | Test accuracy (%) | Reduction time (s) |",
            "Full graph",
        ),
        (
            "zh",
            "| 数据集 | 方法 | 预算 | 缩减后节点数 | 测试准确率（%） | 图缩减耗时（s） |",
            "原图",
        ),
    ],
)
def test_markdown_table_formats_snapshot(language, header, whole_label):
    # The historical table remains exportable independently of the current README.
    cell = dict(
        dataset="ogbn-arxiv",
        test_accuracy_mean=0.75,
        test_accuracy_population_std=0.0125,
        median_reduction_seconds=1.234,
    )
    snapshot = {
        "cells": [
            dict(cell, method="whole", training_node_rate=None, actual_nodes=[1000]),
            dict(cell, method="cgc", training_node_rate="0.001", actual_nodes=[20, 10, 20]),
        ]
    }
    assert markdown_table(snapshot, language).splitlines() == [
        header,
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| ogbn-arxiv | {whole_label} | — | 1,000 | 75.00 ± 1.25 | — |",
        "| ogbn-arxiv | CGC | 0.1% | 10/20 | 75.00 ± 1.25 | 1.23 |",
    ]


def test_frozen_snapshot_figures_exist():
    for dataset in DATASETS:
        for mode in ("light", "dark"):
            assert (ROOT / "figures" / f"tradeoff_{dataset}_{mode}.png").exists()


def test_render_writes_light_and_dark_figures(tmp_path):
    pytest.importorskip("matplotlib")
    snapshot = json.loads(SNAPSHOT.read_text())
    paths = render(snapshot, tmp_path)
    assert sorted(path.name for path in paths) == sorted(
        f"tradeoff_{dataset}_{mode}.png" for dataset in DATASETS for mode in ("light", "dark")
    )
    assert all(path.stat().st_size > 0 for path in paths)

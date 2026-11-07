# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Official split preservation and fail-closed benchmark coverage."""

import json
import io
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from benchmark import expand_datasets as runner
from graphslim.dataset import loader


def example_graph():
    return Data(
        x=torch.randn(6, 3),
        y=torch.tensor([[0], [1], [0], [1], [0], [1]]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
    )


def test_products_preserves_official_split_not_rng_generated(tmp_path, monkeypatch):
    data = example_graph()
    official = {
        "train": torch.tensor([0, 3]),
        "valid": torch.tensor([1, 4]),
        "test": torch.tensor([2, 5]),
    }

    class Dataset:
        num_classes = 2

        def __getitem__(self, index):
            return data.clone()

        def get_idx_split(self):
            return official

    monkeypatch.setattr(loader, "PygNodePropPredDataset", lambda *a, **k: Dataset())
    args = SimpleNamespace(split="fixed", pre_norm=False)
    result = loader.get_dataset("ogbn-products", args, str(tmp_path))
    for name, source in (("train", "train"), ("val", "valid"), ("test", "test")):
        assert torch.equal(getattr(result, f"idx_{name}"), official[source])
    assert result.labels_full.ndim == 1
    assert result.split_provenance["source"] == "ogb.get_idx_split"


def test_ogb_cache_loads_pyg_classes_with_weights_only():
    buffer = io.BytesIO()
    data = example_graph()
    torch.save((data, None), buffer)
    buffer.seek(0)
    with loader.ogb_cache_context():
        loaded, slices = torch.load(buffer, weights_only=True)
    assert torch.equal(loaded.x, data.x) and slices is None


def test_heterophilous_selects_official_column_without_resampling(tmp_path, monkeypatch):
    data = example_graph()
    data.y = data.y.flatten()
    data.train_mask = torch.tensor(
        [[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]], dtype=torch.bool
    )
    data.val_mask = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0], [0, 0], [0, 0]], dtype=torch.bool)
    data.test_mask = torch.tensor(
        [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.bool
    )

    class Dataset:
        num_classes = 2

        def __getitem__(self, index):
            return data.clone()

    monkeypatch.setattr(loader, "HeterophilousGraphDataset", lambda *a, **k: Dataset())
    args = SimpleNamespace(split="fixed", pre_norm=False, dataset_split_id=1)
    result = loader.get_dataset("roman-empire", args, str(tmp_path))
    assert result.idx_train.tolist() == [2, 3]
    assert result.idx_val.tolist() == [0, 1]
    assert result.split_provenance == {"source": "official", "split_id": 1}
    args.dataset_split_id = 2
    with pytest.raises(ValueError, match="Invalid official split"):
        loader.get_dataset("roman-empire", args, str(tmp_path))


def test_expansion_roster_order_unique_and_rates():
    jobs = runner.plan()
    # Twelve datasets: one full graph, four reducers x three rates x three
    # seeds, and one GCGP parity job per dataset.
    assert len(jobs) == 456
    assert {job["method"] for job in jobs if job["kind"] == "reduction"} == {
        "cgc",
        "clustgdd",
        "gcgp",
        "random",
    }
    assert len({job["id"] for job in jobs}) == len(jobs)
    assert [job["stage"] for job in jobs] == sorted(job["stage"] for job in jobs)
    assert len([job for job in jobs if job["kind"] == "parity"]) == 12
    assert {
        job["rate"]
        for job in jobs
        if job["dataset"] == "ogbn-products" and job["kind"] == "reduction"
    } == {"0.0005", "0.001", "0.002"}


def test_reload_sparse_checkpoint_checks_values_and_preserves_layout(tmp_path):
    from graphslim.dataset.utils import save_reduced
    from unittest.mock import Mock

    args = SimpleNamespace(
        save_path=str(tmp_path),
        method="random",
        dataset="cora",
        reduction_rate=0.5,
        seed=1,
        device="cpu",
        logger=Mock(),
        attack=None,
    )
    adjacency = torch.eye(3).to_sparse()
    save_reduced(adjacency, torch.rand(3, 2), torch.tensor([0, 1, 0]), args)
    graph = SimpleNamespace()
    runner.load_checkpoint(graph, args)
    assert graph.adj_syn.layout == torch.sparse_coo
    assert torch.equal(graph.adj_syn.to_dense(), torch.eye(3))
    adjacency.values()[0] = float("nan")
    save_reduced(adjacency, torch.rand(3, 2), torch.tensor([0, 1, 0]), args)
    with pytest.raises(FloatingPointError, match="Non-finite checkpoint"):
        runner.load_checkpoint(graph, args)


def test_export_never_publishes_partial_or_duplicate_seed(tmp_path):
    jobs = runner.plan(["cora"], methods=["cgc"], seeds=[1, 2, 3])
    manifest = {"jobs": jobs, "source_sha256": "frozen", "protocol": {"evaluation_seeds": [0, 1]}}
    job = next(job for job in jobs if job["kind"] == "reduction")
    folder = tmp_path / "jobs" / job["id"]
    folder.mkdir(parents=True)
    (folder / "terminal.json").write_text(
        json.dumps(dict(job=job, source_sha256="frozen", status="complete"))
    )
    (folder / "result.json").write_text(
        json.dumps(dict(job=job, source_sha256="frozen", data={}, evaluations=[]))
    )
    summary = runner.export(tmp_path, manifest)
    cell = next(
        cell for cell in summary["cells"] if cell["method"] == "cgc" and cell["rate"] == job["rate"]
    )
    assert cell["status"] == "pending" and "mean" not in cell
    for other in [
        other for other in jobs if other["kind"] == "reduction" and other["rate"] == job["rate"]
    ]:
        target = tmp_path / "jobs" / other["id"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "terminal.json").write_text(
            json.dumps(dict(job=other, source_sha256="frozen", status="complete"))
        )
        (target / "result.json").write_text(
            json.dumps(
                dict(
                    job=other,
                    source_sha256="frozen",
                    data={},
                    evaluations=[dict(seed=0, test=0.8), dict(seed=0, test=0.8)],
                )
            )
        )
    with pytest.raises(RuntimeError, match="Missing/duplicate"):
        runner.export(tmp_path, manifest)


@pytest.mark.parametrize("dataset", ["photo", "computers"])
def test_amazon_expansion_rates_and_stage(dataset):
    jobs = runner.plan([dataset])
    assert len(jobs) == 38
    assert {job["dataset"] for job in jobs} == {dataset}
    assert {job["stage"] for job in jobs} == {4}
    assert {job["rate"] for job in jobs if job["kind"] == "reduction"} == {"0.005", "0.01", "0.02"}


def test_new_defaults_preserve_frozen_three_stage_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "science_files", lambda: [runner.ROOT / "benchmark/expand_datasets.py"]
    )
    stages = runner.STAGES
    with monkeypatch.context() as patch:
        patch.setattr(runner, "STAGES", stages[:3])
        runner.main(["--run-dir", str(tmp_path), "--initialize-only"])
    paths = [tmp_path / "manifest.json", tmp_path / "source.zip"]
    before = [path.read_bytes() for path in paths]
    runner.main(["--run-dir", str(tmp_path), "--initialize-only"])
    assert [path.read_bytes() for path in paths] == before
    manifest = json.loads(paths[0].read_text())
    assert len(manifest["jobs"]) == 380
    assert len(manifest["stage_order"]) == 3
    with pytest.raises(RuntimeError, match="Frozen roster/protocol differs"):
        runner.main(
            ["--run-dir", str(tmp_path), "--datasets", "photo", "computers", "--initialize-only"]
        )

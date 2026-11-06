"""Exercise actual GraphSAINT files and preprocessing; replace network retrieval only."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import gdown
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from graphslim.dataset import loader
from test_dataset_contracts import raw_graph


def write_saint(path, multilabel=False):
    path.mkdir(parents=True, exist_ok=True)
    raw = raw_graph()
    adjacency = sp.coo_matrix(
        (np.ones(raw.num_edges), raw.edge_index.numpy()), shape=(20, 20)
    ).tocsr()
    sp.save_npz(path / "adj_full.npz", adjacency)
    np.save(path / "feats.npy", raw.x.numpy())
    (path / "role.json").write_text(
        json.dumps({"tr": list(range(12)), "va": list(range(12, 16)), "te": list(range(16, 20))})
    )
    labels = {
        str(i): ([int(i % 2 == 0), int(i % 2 == 1)] if multilabel else 10 + i % 2)
        for i in range(20)
    }
    (path / "class_map.json").write_text(json.dumps(labels))
    return raw


@pytest.mark.parametrize("multilabel", [False, True])
def test_cached_graphsaint_files_roundtrip_without_downloading(tmp_path, monkeypatch, multilabel):
    raw = write_saint(tmp_path / "ogbn_arxiv/raw", multilabel)

    def unexpected(**kwargs):
        pytest.fail("Complete cache must not trigger a download")

    monkeypatch.setattr(gdown, "download_folder", unexpected)
    value = loader.DataGraphSAINT(str(tmp_path), "ogbn-arxiv")
    assert value[0] is value and value.get(0) is value
    assert value.num_nodes == 20 and value.nclass == 2
    np.testing.assert_array_equal(value.feat_full, raw.x.numpy())
    np.testing.assert_array_equal(value.adj_full.toarray(), value.adj_full.T.toarray())
    assert value.train_mask.sum() == 12 and value.test_mask.sum() == 4
    if multilabel:
        torch.testing.assert_close(value.labels_full.sum(1), torch.ones(20))
    else:
        assert torch.equal(value.labels_full, raw.y)


@pytest.mark.parametrize("delivery", ["nested", "files", "already_in_destination"])
def test_download_delivery_layouts_are_flattened_and_loaded(tmp_path, monkeypatch, delivery):
    source = tmp_path / "remote"
    write_saint(source)

    def download(**kwargs):
        target = Path(kwargs["output"])
        if delivery == "nested":
            nested = target / "downloaded"
            shutil.copytree(source, nested)
            return [None, str(nested)]
        if delivery == "files":
            return [str(p) for p in source.iterdir()]
        for p in source.iterdir():
            shutil.copy2(p, target / p.name)
        return str(target)

    monkeypatch.setattr(gdown, "download_folder", download)
    result = loader.DataGraphSAINT(str(tmp_path / "data"), "ogbn-arxiv")
    assert result.labels_full.tolist() == [i % 2 for i in range(20)]
    assert not (tmp_path / "data/ogbn_arxiv/raw/downloaded").exists()


@pytest.mark.parametrize("download_error", [False, True])
def test_failed_download_uses_real_ogb_conversion_and_persists_reloadable_cache(
    tmp_path, monkeypatch, download_error
):
    import ogb.nodeproppred

    raw = raw_graph()
    raw.y = raw.y[:, None]

    class Dataset:
        def __getitem__(self, index):
            return raw.clone()

        def get_idx_split(self):
            return {
                "train": torch.arange(12),
                "valid": torch.arange(12, 16),
                "test": torch.arange(16, 20),
            }

    def unavailable(**kwargs):
        if download_error:
            raise OSError("Simulated network outage")
        return []

    monkeypatch.setattr(gdown, "download_folder", unavailable)
    monkeypatch.setattr(ogb.nodeproppred, "PygNodePropPredDataset", lambda **kw: Dataset())
    result = loader.DataGraphSAINT(str(tmp_path), "ogbn-arxiv")
    assert result.nclass == 2
    torch.testing.assert_close(result.labels_full, raw.y[:, 0])
    cached = loader.DataGraphSAINT(str(tmp_path), "ogbn-arxiv")
    np.testing.assert_array_equal(cached.feat_full, result.feat_full)
    assert (cached.adj_full != result.adj_full).nnz == 0
    assert cached.idx_train == list(range(12))


def test_unavailable_non_ogb_dataset_reports_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gdown, "download_folder", lambda **kw: None)
    with pytest.raises(RuntimeError, match="Unable to prepare dataset files"):
        loader.DataGraphSAINT(str(tmp_path), "missing")


def test_arxiv_public_loader_and_large_batches_use_actual_saint_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_saint(tmp_path / "data/ogbn_arxiv/raw")
    args = SimpleNamespace(split="fixed", pre_norm=False)
    graph = loader.get_dataset("OGBN_ARXIV", args, "data")
    assert graph.nclass == 40 and graph.feat_train.shape == (12, 3)
    torch.testing.assert_close(graph.feat_train.mean(0), torch.zeros(3), atol=1e-6, rtol=0)
    batches = loader.LargeDataLoader("ogbn-arxiv", batch_size=4)
    batches.split_batch()
    assert batches.properties() == (3, 12, 40, 3, 20)
    seen = []
    for cluster in range(3):
        features, labels, adjacency = batches.get_batch(cluster)
        indices = torch.where(torch.as_tensor(batches.batch_labels) == cluster)[0]
        seen.extend(indices.tolist())
        torch.testing.assert_close(features, batches.split_feat[indices])
        assert torch.isfinite(features).all() and adjacency.shape == (len(labels), len(labels))
    assert sorted(seen) == list(range(12))


def test_safe_cache_context_handles_unavailable_safe_globals(monkeypatch):
    monkeypatch.setattr(loader, "torch_serialization", SimpleNamespace())
    with loader.ogb_cache_context():
        assert True


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("depth", [1, 3, 4, 5])
def test_real_class_neighbor_samples_keep_seed_nodes_inside_training_view(setting, depth):
    from test_legacy_reducer_integration import graph_fixture

    graph = graph_fixture()
    adjacency = graph.adj_full if setting == "trans" else graph.adj_train
    adj = loader.SparseTensor.from_scipy(adjacency)
    args = SimpleNamespace(setting=setting, nlayers=depth, dataset="reddit")
    for label in range(2):
        batch_size, node_ids, adjs = graph.retrieve_class_sampler(label, adj, args, num=3)
        labels = graph.labels_full if setting == "trans" else graph.labels_train
        assert batch_size == 3 and labels[node_ids[:batch_size]].eq(label).all()
        if setting == "trans":
            assert torch.isin(node_ids[:batch_size], graph.idx_train).all()
        blocks = [adjs] if depth == 1 else adjs
        assert len(blocks) == depth


def test_large_loader_accepts_single_node_batches(monkeypatch):
    import torch_geometric.datasets

    class Dataset:
        num_classes = 2

        def __getitem__(self, index):
            return raw_graph()

    monkeypatch.setattr(torch_geometric.datasets, "Flickr", lambda **kw: Dataset())
    batches = loader.LargeDataLoader("flickr", batch_size=1)
    batches.split_batch()
    assert 1 <= batches.k <= batches.n_split
    assert len(batches.batch_labels) == 12

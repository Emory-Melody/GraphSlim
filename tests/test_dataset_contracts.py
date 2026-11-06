"""Mock dataset retrieval only; execute real split, preprocessing, and graph views."""

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset import loader
from graphslim.dataset.utils import splits


def raw_graph(with_masks=True):
    nodes = torch.arange(20)
    data = Data(
        x=torch.arange(60).reshape(20, 3).float() + 1,
        y=nodes % 2,
        edge_index=torch.stack([nodes.repeat(2), torch.cat([nodes.roll(1), nodes.roll(-1)])]),
    )
    if with_masks:
        data.train_mask = nodes < 12
        data.val_mask = (nodes >= 12) & (nodes < 16)
        data.test_mask = nodes >= 16
    return data


@pytest.mark.parametrize(
    "dataset,constructor",
    [
        ("CORA", "Planetoid"),
        ("Cora_ML", "CitationFull"),
        ("cs", "Coauthor"),
        ("photo", "Amazon"),
        ("flickr", "Flickr"),
        ("reddit", "Reddit2"),
    ],
)
def test_loader_dispatch_preserves_splits_and_preprocesses_training_only(
    tmp_path, monkeypatch, dataset, constructor
):
    raw = raw_graph()

    class Dataset:
        num_classes = 2

        def __getitem__(self, index):
            return raw.clone()

    monkeypatch.setattr(loader, constructor, lambda *a, **kw: Dataset())
    args = SimpleNamespace(split="fixed", pre_norm=True)
    result = loader.get_dataset(dataset, args, str(tmp_path))
    assert torch.equal(result.idx_train, torch.arange(12))
    assert torch.equal(result.idx_val, torch.arange(12, 16))
    assert torch.equal(result.idx_test, torch.arange(16, 20))
    assert result.adj_train.shape == (12, 12)
    if dataset.lower() in {"flickr", "reddit"}:
        torch.testing.assert_close(result.feat_train.mean(0), torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(result.feat_train.std(0, unbiased=False), torch.ones(3))
    elif dataset.lower() == "cora":
        torch.testing.assert_close(result.feat_full.sum(1), torch.ones(20))
    else:
        torch.testing.assert_close(result.feat_full, raw.x)
    assert result.to("cpu") is result
    result.nclass = 2
    for class_id in range(2):
        selected = result.retrieve_class(class_id, num=3)
        assert len(selected) == 3 and (result.labels_train[selected] == class_id).all()
    result.labels_syn = torch.ones(2)
    result.reset()
    assert result.labels_syn is result.feat_syn is result.adj_syn is None
    assert result.samplers is result.class_dict2 is None


@pytest.mark.parametrize(
    "strategy,counts", [("default", [16, 2, 2]), ("few", [10, 10, 0]), ("random", [20, 0, 0])]
)
def test_generated_splits_are_disjoint_class_balanced_and_deterministic(strategy, counts):
    torch.manual_seed(7)
    first = splits(raw_graph(False), strategy)
    torch.manual_seed(7)
    second = splits(raw_graph(False), strategy)
    masks = [first.train_mask, first.val_mask, first.test_mask]
    assert [int(m.sum()) for m in masks] == counts
    assert torch.equal(torch.stack(masks).sum(0), torch.ones(20, dtype=torch.long))
    for name in ["train", "val", "test"]:
        assert torch.equal(getattr(first, name + "_mask"), getattr(second, name + "_mask"))
    assert first.y[first.train_mask].bincount().tolist() == [counts[0] // 2] * 2


def test_loader_rejects_unknown_dataset_before_io(tmp_path):
    with pytest.raises(ValueError, match="not recognized"):
        loader.get_dataset("not-a-dataset", SimpleNamespace(), str(tmp_path))


@pytest.mark.parametrize("dataset", ["flickr", "reddit"])
def test_large_loader_batches_partition_training_nodes_and_keep_induced_edges(monkeypatch, dataset):
    import torch_geometric.datasets as datasets

    raw = raw_graph()

    class Dataset:
        num_classes = 2

        def __getitem__(self, index):
            return raw.clone()

    monkeypatch.setattr(
        datasets, "Flickr" if dataset == "flickr" else "Reddit2", lambda **kw: Dataset()
    )
    value = loader.LargeDataLoader(dataset, batch_size=4)
    value.split_batch()
    seen = []
    for batch_id in range(value.k.item()):
        features, labels, adjacency = value.get_batch(batch_id)
        indices = torch.where(torch.tensor(value.batch_labels) == batch_id)[0]
        seen.extend(indices.tolist())
        torch.testing.assert_close(features, value.split_feat[indices])
        assert torch.equal(labels, raw.y[value.split_idx[indices]])
        source = value.Adj.to_dense()[value.split_idx[indices]][:, value.split_idx[indices]]
        torch.testing.assert_close(adjacency.to_dense(), source)
    assert sorted(seen) == list(range(12))
    assert value.properties() == (value.k, 12, 2, 3, 20)
    torch.testing.assert_close(
        value.GCF(torch.eye(3).to_sparse(), torch.ones(3, 2)), torch.ones(3, 2)
    )

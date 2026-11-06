"""Dataset dispatch, official splits, samplers, and GraphSAINT preparation without downloads."""

import sys
import types
from types import SimpleNamespace

import gdown
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset import loader
from graphslim.dataset.utils import splits
from test_dataset_contracts import raw_graph
from test_graphsaint_io import write_saint


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def args(split="fixed", **extra):
    return SimpleNamespace(split=split, pre_norm=True, **extra)


@pytest.mark.parametrize("name", ["ogbn-products", "OGBN_PROTEINS", "ogbn-papers100M"])
def test_ogb_datasets_use_official_index_split(tmp_path, monkeypatch, name):
    raw = raw_graph(with_masks=False)
    raw.y = raw.y[:, None]
    calls = []

    class Dataset:
        num_classes = 2

        def __init__(self, dataset_name, root):
            calls.append((dataset_name, root))

        def __getitem__(self, index):
            return raw

        def get_idx_split(self):
            return {
                "train": torch.tensor([0, 2, 4, 6, 8, 1, 3]),
                "valid": torch.tensor([5, 7]),
                "test": torch.arange(9, 20),
            }

    monkeypatch.setattr(loader, "PygNodePropPredDataset", Dataset)
    graph = loader.get_dataset(name, args(), str(tmp_path))
    assert calls == [(name.lower().replace("_", "-"), str(tmp_path))]
    assert graph.idx_train.tolist() == [0, 1, 2, 3, 4, 6, 8]
    assert graph.idx_val.tolist() == [5, 7]
    assert graph.labels_full.shape == (20,)
    assert graph.split_provenance == {"source": "ogb.get_idx_split"}
    assert graph.nclass == 2


def heterophilous(masks_2d=True):
    raw = raw_graph(with_masks=False)
    nodes = torch.arange(20)
    columns = []
    for shift in range(3):
        rotated = (nodes + 4 * shift) % 20
        columns.append((rotated < 12, (rotated >= 12) & (rotated < 16), rotated >= 16))
    for position, key in enumerate(("train", "val", "test")):
        stacked = torch.stack([column[position] for column in columns], dim=1)
        setattr(raw, f"{key}_mask", stacked if masks_2d else stacked[:, 0])
    return raw


@pytest.mark.parametrize("split_id", [0, 2])
def test_heterophilous_graphs_select_requested_official_split(tmp_path, monkeypatch, split_id):
    raw = heterophilous()

    class Dataset:
        num_classes = 2

        def __init__(self, root, name):
            assert name == "roman-empire"

        def __getitem__(self, index):
            return raw

    monkeypatch.setattr(loader, "HeterophilousGraphDataset", Dataset)
    graph = loader.get_dataset("Roman_Empire", args(dataset_split_id=split_id), str(tmp_path))
    expected = torch.where(((torch.arange(20) + 4 * split_id) % 20) < 12)[0]
    assert graph.idx_train.tolist() == expected.tolist()
    assert graph.split_provenance == {"source": "official", "split_id": split_id}


@pytest.mark.parametrize("split_id,masks_2d", [(3, True), (-1, True), (0, False)])
def test_heterophilous_graphs_reject_invalid_split_ids(tmp_path, monkeypatch, split_id, masks_2d):
    raw = heterophilous(masks_2d)

    class Dataset:
        num_classes = 2

        def __init__(self, root, name):
            pass

        def __getitem__(self, index):
            return raw

    monkeypatch.setattr(loader, "HeterophilousGraphDataset", Dataset)
    with pytest.raises(ValueError, match="Invalid official split ID"):
        loader.get_dataset("amazon-ratings", args(dataset_split_id=split_id), str(tmp_path))


def test_fraud_datasets_require_the_optional_dgl_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "dgl", None)
    monkeypatch.setitem(sys.modules, "dgl.data", None)
    with pytest.raises(ImportError, match="graphslim\\[dgl\\]"):
        loader.get_dataset("yelp", args(), str(tmp_path))


def test_fraud_datasets_convert_dgl_graph_and_create_classwise_split(tmp_path, monkeypatch):
    raw = raw_graph(with_masks=False)
    raw.num_classes = 2
    seen = {}

    class FraudDataset:
        def __init__(self, name, raw_dir):
            seen["init"] = (name, raw_dir)

        def __getitem__(self, index):
            return "dgl-graph"

    def from_dgl(graph, name, hetero):
        seen["convert"] = (graph, name, hetero)
        return raw

    package = types.ModuleType("dgl")
    module = types.ModuleType("dgl.data")
    module.FraudDataset = FraudDataset
    package.data = module
    monkeypatch.setitem(sys.modules, "dgl", package)
    monkeypatch.setitem(sys.modules, "dgl.data", module)
    monkeypatch.setattr(loader, "from_dgl", from_dgl)
    torch.manual_seed(0)
    graph = loader.get_dataset("Amazon", args(split="default"), str(tmp_path))
    assert seen == {
        "init": ("amazon", str(tmp_path)),
        "convert": ("dgl-graph", "amazon", False),
    }
    assert graph.nclass == 2
    assert (len(graph.idx_train), len(graph.idx_val), len(graph.idx_test)) == (16, 2, 2)
    for index in (graph.idx_train, graph.idx_val, graph.idx_test):
        assert torch.bincount(graph.labels_full[index]).tolist() == [len(index) // 2] * 2


def wrapped_graph():
    graph = loader.TransAndInd(splits(raw_graph()), "cora", norm=False)
    graph.nclass = 2
    return graph


def test_wrapper_views_class_sampling_device_move_and_reset():
    graph = wrapped_graph()
    assert graph.adj_train.shape == (12, 12)
    assert (graph.adj_val != graph.adj_full[12:16][:, 12:16]).nnz == 0
    np.random.seed(0)
    chosen = graph.retrieve_class(1, num=4)
    assert len(chosen) == 4 and (graph.labels_train[chosen] == 1).all()
    assert sorted(graph.retrieve_class(0, num=100).tolist()) == list(range(0, 12, 2))
    assert graph.to("cpu") is graph and graph.feat_test.device.type == "cpu"
    graph.samplers, graph.class_dict2 = ["stale"], {"stale": 1}
    graph.reset()
    assert graph.samplers is None and graph.class_dict2 is None
    assert graph.feat_syn is None and graph.adj_syn is None and graph.labels_syn is None


def test_saint_style_inputs_populate_the_same_views():
    pyg = wrapped_graph()
    saint = SimpleNamespace(
        adj_full=pyg.adj_full,
        feat_full=pyg.feat_full,
        labels_full=pyg.labels_full,
        num_nodes=20,
        train_mask=pyg.train_mask,
        val_mask=pyg.val_mask,
        test_mask=pyg.test_mask,
        idx_train=pyg.idx_train,
        idx_val=pyg.idx_val,
        idx_test=pyg.idx_test,
    )
    graph = loader.TransAndInd(saint, "cs", norm=False)
    assert torch.equal(graph.x, pyg.feat_full) and torch.equal(graph.y, pyg.labels_full)
    assert sorted(map(tuple, graph.edge_index.T.tolist())) == sorted(
        map(tuple, pyg.edge_index.T.tolist())
    )
    assert graph.sparse_adj.nnz() == pyg.adj_full.nnz


@pytest.mark.parametrize("dataset", ["reddit", "cora"])
def test_two_layer_class_sampler_uses_dataset_specific_fanout(dataset):
    graph = wrapped_graph()
    adj = loader.SparseTensor.from_scipy(graph.adj_full)
    options = SimpleNamespace(setting="trans", nlayers=2, dataset=dataset)
    np.random.seed(0)
    batch_size, node_ids, blocks = graph.retrieve_class_sampler(1, adj, options, num=2)
    assert batch_size == 2 and len(blocks) == 2
    assert graph.labels_full[node_ids[:2]].eq(1).all()
    assert graph.samplers[0].sizes == ([15, 8] if dataset == "reddit" else [10, 5])


def test_safe_global_registration_tolerates_missing_serialization_support(tmp_path, monkeypatch):
    import ogb.nodeproppred

    raw = raw_graph()
    raw.y = raw.y[:, None]

    class Dataset:
        def __init__(self, name, root):
            assert root.endswith("ogb_cache")

        def __getitem__(self, index):
            return raw

        def get_idx_split(self):
            return {
                "train": torch.arange(12),
                "valid": torch.arange(12, 16),
                "test": torch.arange(16, 20),
            }

    registered = []
    monkeypatch.setattr(ogb.nodeproppred, "PygNodePropPredDataset", Dataset)
    monkeypatch.setattr(gdown, "download_folder", lambda **kw: ["not-a-path", 7])
    for value in (None, SimpleNamespace(), SimpleNamespace(add_safe_globals=registered.extend)):
        monkeypatch.setattr(loader, "torch_serialization", value)
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        graph = loader.DataGraphSAINT(str(root), "ogbn-arxiv")
        assert graph.idx_val == [12, 13, 14, 15]
    for name in ("DataEdgeAttr", "DataTensorAttr", "GlobalStorage"):
        monkeypatch.setattr(loader, name, None)
    loader.DataGraphSAINT(str(tmp_path / "none"), "ogbn-arxiv")
    assert len(registered) == 3


def test_arxiv_fallback_that_writes_nothing_reports_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gdown, "download_folder", lambda **kw: [])
    monkeypatch.setattr(loader.DataGraphSAINT, "_prepare_ogbn_arxiv_from_ogb", lambda *a: None)
    with pytest.raises(RuntimeError, match="adj_full.npz"):
        loader.DataGraphSAINT(str(tmp_path), "ogbn-arxiv")


def test_partial_download_completes_cache_and_keeps_existing_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "ogbn_arxiv/raw"
    write_saint(raw_dir, multilabel=True)
    (raw_dir / "role.json").unlink()
    source = tmp_path / "remote"
    write_saint(source, multilabel=True)

    def download(**kwargs):
        return str(source / "role.json")

    monkeypatch.setattr(gdown, "download_folder", download)
    graph = loader.DataGraphSAINT(str(tmp_path), "ogbn-arxiv")
    assert graph.nclass == 2 and graph.labels_full.shape == (20, 2)
    assert (raw_dir / "role.json").exists() and not (source / "role.json").exists()


def test_ogb_cache_context_is_a_context_manager():
    with loader.ogb_cache_context():
        value = Data(x=torch.ones(1, 1))
    assert value.num_nodes == 1


def test_ogb_partition_loader_placeholder_is_an_empty_module():
    module = loader.OgbDataLoader()
    assert isinstance(module, torch.nn.Module)
    assert list(module.parameters()) == [] and list(module.children()) == []

"""Sparsification thresholds, split creation, checkpoint IO, and tree-mining helpers."""

import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset import utils
from test_lightweight_methods import tiny_graph


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(("info", message))

    def warning(self, message, *args):
        self.messages.append(("warning", message))


def checkpoint_args(tmp_path, **extra):
    values = dict(
        save_path=str(tmp_path),
        method="gcond",
        attack=None,
        dataset="tiny",
        reduction_rate=0.5,
        seed=1,
        setting="trans",
        device="cpu",
        logger=RecordingLogger(),
    )
    values.update(extra)
    return SimpleNamespace(**values)


ADJ = torch.tensor([[0.05, 0.2, 0.6], [0.2, 0.0, 0.3], [0.6, 0.3, 0.9]])


@pytest.mark.parametrize(
    "model,method,dataset,threshold,expected",
    [
        ("GAT", "cgc", "cora", None, 0.0),
        ("GAT", "gcond", "cora", None, 0.5),
        ("GAT", "doscond", "flickr", None, 0.1),
        ("GAT", "msgc", "cora", 0.25, 0.25),
        ("GAT", "random", "cora", None, 0.5),
        ("GCN", "gcond", "cora", 0.25, 0.25),
        ("GCN", "msgc", "cora", 0.25, 0.0),
        ("GCN", "random", "cora", 0.25, 0.0),
    ],
)
def test_sparsify_applies_the_model_and_method_threshold(
    model, method, dataset, threshold, expected, capsys
):
    args = SimpleNamespace(method=method, dataset=dataset, threshold=threshold)
    result = utils.sparsify(model, ADJ.clone(), args, verbose=True)
    torch.testing.assert_close(result, torch.where(ADJ < expected, 0.0, ADJ))
    assert ("Sparsity after truncating" in capsys.readouterr().out) == (expected > 0)


def test_mlp_sparsification_replaces_topology_with_identity():
    args = SimpleNamespace(method="gcond", dataset="cora", threshold=0.1)
    torch.testing.assert_close(utils.sparsify("MLP", ADJ.clone(), args), torch.eye(3))


def test_index_mask_supports_vectors_and_matrix_shapes():
    assert utils.index2mask([0, 2], 4).tolist() == [True, False, True, False]
    mask = utils.index2mask((torch.tensor([0, 1]), torch.tensor([1, 0])), (2, 2))
    assert mask.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize("exp,sizes", [("random", (40, 60, 20)), ("few", (10, 10, 100))])
def test_generated_split_protocols_have_exact_class_counts(exp, sizes):
    nodes = torch.arange(120)
    data = Data(x=torch.ones(120, 1), y=(nodes % 2)[:, None], num_nodes=120)
    torch.manual_seed(0)
    result = utils.splits(data, exp)
    observed = (len(result.idx_train), len(result.idx_val), len(result.idx_test))
    assert observed == sizes
    assert result.y.shape == (120,)
    combined = torch.cat([result.idx_train, result.idx_val, result.idx_test])
    assert sorted(combined.tolist()) == list(range(120))


def test_saint_style_split_uses_labels_full_and_existing_masks():
    data = SimpleNamespace(
        labels_full=torch.tensor([0, 1, 2, 0]),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
    )
    result = utils.splits(data)
    assert result.idx_train.tolist() == [0, 1]
    assert result.idx_val.tolist() == [2] and result.idx_test.tolist() == [3]


def test_attacked_flickr_checkpoints_use_the_corrupt_graph_tree(tmp_path):
    args = checkpoint_args(tmp_path, dataset="flickr", attack="random_adj")
    feat, labels = torch.ones(2, 3), torch.tensor([0, 1])
    utils.save_reduced(torch.eye(2), feat, labels, args)
    folder = tmp_path / "corrupt_graph/random_adj/reduced_graph/gcond"
    assert sorted(p.name for p in folder.iterdir()) == [
        "adj_flickr_0.5_1.pt",
        "feat_flickr_0.5_1.pt",
        "label_flickr_0.5_1.pt",
    ]
    adj, loaded_feat, loaded_labels = utils.load_reduced(args)
    torch.testing.assert_close(adj, torch.eye(2))
    torch.testing.assert_close(loaded_feat, feat)
    assert loaded_labels.tolist() == [0, 1]
    assert args.logger.messages[0][0] == "info"


def test_missing_checkpoints_raise_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="--eval_whole"):
        utils.load_reduced(checkpoint_args(tmp_path))


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_unreadable_features_fall_back_to_original_view(tmp_path, setting):
    graph = tiny_graph()
    args = checkpoint_args(tmp_path, setting=setting)
    utils.save_reduced(labels_syn=torch.tensor([1, 0]), args=args)
    folder = tmp_path / "reduced_graph/gcond"
    (folder / "feat_tiny_0.5_1.pt").write_bytes(b"not a checkpoint")
    adj, feat, labels = utils.load_reduced(args, graph)
    expected = graph.feat_full if setting == "trans" else graph.feat_train
    assert feat is expected
    torch.testing.assert_close(adj, torch.eye(len(expected)))
    assert labels.tolist() == [1, 0]
    warnings = [m for kind, m in args.logger.messages if kind == "warning"]
    assert len(warnings) == 2 and "synthetic features" in warnings[0]


@pytest.mark.parametrize(
    "device,available,count,expected",
    [
        ("cuda:0", False, 0, "cpu"),
        ("cuda:3", True, 0, "cpu"),
        ("cuda:3", True, 1, "cuda:0"),
        ("cuda:x", True, 1, "cuda:0"),
        ("cuda", True, 1, "cuda:0"),
    ],
)
def test_cuda_requests_resolve_and_fall_back_without_a_gpu(
    tmp_path, monkeypatch, device, available, count, expected
):
    args = checkpoint_args(tmp_path)
    utils.save_reduced(torch.eye(2), torch.ones(2, 3), torch.tensor([0, 1]), args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    args.device = device
    adj, feat, labels = utils.load_reduced(args)
    assert args.device == expected
    # Without real CUDA the tensors stay on CPU after the attempted move.
    assert adj.device.type == feat.device.type == labels.device.type == "cpu"
    torch.testing.assert_close(adj, torch.eye(2))
    warnings = [m for kind, m in args.logger.messages if kind == "warning"]
    if device == "cuda:3":
        assert any("unavailable" in m for m in warnings)
    if expected != "cpu":
        assert any("failed to move" in m for m in warnings)


def test_full_size_labels_return_reloaded_graph_without_sparsification(tmp_path):
    graph = tiny_graph()
    args = checkpoint_args(tmp_path, threshold=0.5)
    adjacency = torch.full((16, 16), 0.1)
    utils.save_reduced(adjacency, graph.feat_train, graph.labels_train, args)
    feat, adj, labels = utils.get_syn_data(graph, args, "GCN")
    torch.testing.assert_close(adj, adjacency)
    torch.testing.assert_close(labels, graph.labels_train)


@pytest.mark.parametrize("sparse", [False, True])
def test_reduced_graph_is_densified_and_thresholded_for_downstream_model(tmp_path, sparse):
    graph = tiny_graph()
    args = checkpoint_args(tmp_path, threshold=0.5)
    dense = torch.tensor([[1.0, 0.2, 0.0], [0.2, 1.0, 0.7], [0.0, 0.7, 1.0]])
    stored = dense.to_sparse() if sparse else dense
    utils.save_reduced(stored, torch.ones(3, 8), torch.tensor([0, 1, 1]), args)
    feat, adj, labels = utils.get_syn_data(graph, args, "GCN", verbose=False)
    assert adj.layout == torch.strided
    torch.testing.assert_close(adj, torch.where(dense < 0.5, 0.0, dense))
    assert feat.shape == (3, 8) and labels.tolist() == [0, 1, 1]


def test_explicit_device_and_duck_typed_trees_without_edge_features():
    first = SimpleNamespace(
        x=torch.ones(2, 3), edge_index=torch.tensor([[1], [0]]), y=torch.tensor([1])
    )
    second = SimpleNamespace(
        x=torch.zeros(3, 3), edge_index=torch.tensor([[1, 2], [0, 0]]), y=torch.tensor([1])
    )
    result = utils.disjointed_union([first, second], class_=1, device=torch.device("cpu"))
    assert result.edge_index.tolist() == [[1, 3, 4], [0, 2, 2]]
    torch.testing.assert_close(result.x, torch.cat([first.x, second.x]))
    assert result.y == 1 and "edge_attr" not in result
    with pytest.raises(AssertionError, match="same device"):
        utils.disjointed_union([first, second], device=torch.device("meta"))


def test_dict_dataloader_drops_length_option_before_batching():
    graph = Data(x=torch.ones(2, 1), edge_index=torch.tensor([[1], [0]]), y=torch.tensor([0]))
    dataset = {0: [{0: {"freq": 1, "data": graph}}], 1: [{1: {"freq": 1, "data": graph}}]}
    value = utils.get_dataloader(dataset, dataset_len=4, batch_size=1)
    assert isinstance(value.dataset, utils.myDataset)
    assert value.batch_size == 1
    assert all(batch.num_graphs == 1 for batch in value)


def test_frequent_pattern_mining_uses_each_class_threshold(monkeypatch):
    calls = []

    def find_frequent_patterns(transactions, threshold):
        calls.append((transactions, threshold))
        return {("a",): len(transactions)}

    module = types.ModuleType("pyfpgrowth")
    module.find_frequent_patterns = find_frequent_patterns
    monkeypatch.setitem(sys.modules, "pyfpgrowth", module)
    patterns = utils.pyfpgrowth_wrapper({0: [["a"], ["a", "b"]], 1: [["c"]]}, {0: 2, 1: 1})
    assert patterns == {0: {("a",): 2}, 1: {("a",): 1}}
    assert calls == [([["a"], ["a", "b"]], 2), ([["c"]], 1)]


def test_legacy_parser_rejects_a_root_without_children_marker():
    with pytest.raises(AssertionError, match="Parsing failed"):
        utils.parse_canonical_label_bak("7$$")


def test_logger_messages_from_save_reduced_name_the_adjacency_file(tmp_path, caplog):
    args = checkpoint_args(tmp_path, logger=logging.getLogger("cov-data-utils"))
    with caplog.at_level(logging.INFO, logger="cov-data-utils"):
        utils.save_reduced(feat_syn=torch.ones(1, 1), args=args)
    assert "adj_tiny_0.5_1.pt" in caplog.text
    assert not (tmp_path / "reduced_graph/gcond/adj_tiny_0.5_1.pt").exists()


def test_all_corrupt_components_fall_back_to_original_graph(tmp_path):
    graph = tiny_graph()
    args = checkpoint_args(tmp_path, setting="ind")
    utils.save_reduced(torch.eye(2), torch.ones(2, 3), torch.tensor([0, 1]), args)
    for path in (tmp_path / "reduced_graph/gcond").iterdir():
        path.write_bytes(b"corrupt")
    adj, feat, labels = utils.load_reduced(args, graph)
    assert feat is graph.feat_train and labels is graph.labels_train
    torch.testing.assert_close(adj, torch.eye(16))
    assert ("info", "no file found, use original graph instead") in args.logger.messages

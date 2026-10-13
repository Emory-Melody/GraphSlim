"""Released SimGC citation profiles and operator-level port regression checks."""

from copy import copy
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.nn import GCNConv

from benchmark import paper_hpo as hpo
from graphslim.condensation.simgc import SimGC
from graphslim.config import get_args
from graphslim.evaluation import Evaluator
from graphslim.models import GCN


def tiny_graph():
    features, labels = torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])
    return SimpleNamespace(
        nclass=2,
        feat_full=features,
        feat_train=features,
        feat_val=features,
        feat_syn=features,
        labels_train=labels,
        labels_val=labels,
        labels_syn=labels,
        adj_syn=torch.eye(2),
        adj_train=torch.eye(2),
        adj_val=torch.eye(2),
        idx_val=torch.arange(2),
    )


def citation_args(tmp_path, dataset="cora", extra=()):
    return get_args(
        ["-M", "simgc", "-D", dataset, "-G", "-1", "--save_path", str(tmp_path), *extra]
    )


@pytest.mark.parametrize("activation", [None, "none", "linear", "tanh", "sigmoid", "relu"])
@pytest.mark.parametrize("adj_list", [False, True])
def test_gcn_honors_activation_in_dense_and_sampled_paths(tmp_path, activation, adj_list):
    args = citation_args(tmp_path)
    args.activation, args.hidden = activation, 3
    model = GCN(1, 3, 2, args, mode="cross")
    model.eval()
    x, adj = torch.tensor([[-1.0], [2.0]]), torch.eye(2)
    hidden = model.layers[0](x, adj)
    if activation == "tanh":
        hidden = hidden.tanh()
    elif activation == "sigmoid":
        hidden = hidden.sigmoid()
    elif activation == "relu":
        hidden = hidden.relu()
    expected = torch.log_softmax(model.layers[1](hidden, adj).reshape(2, 2), dim=1)
    actual = model(x, [adj, adj] if adj_list else adj)
    assert torch.allclose(actual, expected)
    assert model.with_relu == (activation not in (None, "none", "linear"))


@pytest.mark.parametrize("dataset", ["cora", "citeseer"])
def test_citation_author_profiles_do_not_leak_between_stages(tmp_path, dataset):
    args = citation_args(tmp_path, dataset)
    assert (args.hidden, args.nlayers, args.lr, args.dropout, args.weight_decay) == (
        1024,
        2,
        0.001,
        0,
        0,
    )
    assert args.activation is None and args.lr_decay is False
    assert args.pre_norm is False
    assert (args.eval_epochs, args.val_interval, args.run_inter_eval) == (3001, 100, 1)
    assert args.teacher_epochs == 1001 and args.teacher_val_interval == 50
    assert args.checkpoints == list(range(100, 1501, 100))
    args.hidden, args.dropout, args.lr, args.nlayers = 64, 0.5, 0.01, 4
    reducer = SimGC.__new__(SimGC)
    reducer.args = args
    validation = reducer.validation_model_args()
    assert (validation.hidden, validation.lr, validation.dropout, validation.nlayers) == (
        2048,
        0.001,
        0,
        2,
    )
    assert validation.eval_epochs == 3000 and validation.activation is None
    assert (args.hidden, args.lr, args.dropout, args.nlayers) == (64, 0.01, 0.5, 4)


def test_explicit_cli_can_shorten_both_stages_and_change_activation(tmp_path):
    args = citation_args(
        tmp_path, extra=("--hidden", "4", "--eval_epochs", "2", "--activation", "tanh")
    )
    reducer = SimGC.__new__(SimGC)
    reducer.args = args
    validation = reducer.validation_model_args()
    assert (args.hidden, args.eval_epochs, args.activation) == (4, 2, "tanh")
    assert (validation.hidden, validation.eval_epochs, validation.activation) == (4, 2, "tanh")


def test_citation_loader_keeps_raw_features_without_changing_other_methods(tmp_path):
    from torch_geometric.data import Data
    from graphslim.dataset.loader import TransAndInd
    from graphslim.dataset.utils import splits

    raw = torch.tensor([[1.0, 2.0], [3.0, 1.0], [1.0, 1.0]])
    data = Data(
        x=raw,
        y=torch.tensor([0, 1, 0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        train_mask=torch.tensor([True, False, False]),
        val_mask=torch.tensor([False, True, False]),
        test_mask=torch.tensor([False, False, True]),
    )
    data = splits(data)
    args = citation_args(tmp_path)
    graph = TransAndInd(data, "cora", norm=args.pre_norm)
    assert torch.equal(graph.feat_full, raw)
    assert torch.equal(graph.feat_train, raw[:1])
    normalized = TransAndInd(data, "cora", norm=True)
    assert torch.allclose(normalized.feat_full.sum(dim=1), torch.ones(3))


def test_simgc_uses_author_per_class_ceiling_for_uneven_classes():
    reducer = SimGC.__new__(SimGC)
    reducer.args = SimpleNamespace(reduction_rate=0.5)
    graph = SimpleNamespace(labels_train=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]))
    labels = reducer.generate_labels_syn(graph)
    assert labels.tolist() == [0, 0, 1, 1, 1]
    assert graph.num_class_dict == {0: 2, 1: 3}
    assert reducer.syn_class_indices == {0: [0, 2], 1: [2, 5]}


def test_citation_grid_includes_author_baseline_and_legacy_grid_is_frozen(tmp_path):
    baseline = {"hidden": 1024, "lr": 0.001, "nlayers": 2, "weight_decay": 0, "dropout": 0}
    assert len(hpo.gcn_grid()) == 108
    assert len(hpo.gcn_grid("cora")) == len(hpo.gcn_grid("citeseer")) == 109
    assert hpo.gcn_grid("cora")[-1] == baseline
    legacy = {"gcn_grid": [{"hidden": 7}]}
    assert hpo.dataset_gcn_grid(legacy, "cora") == [{"hidden": 7}]
    args = hpo.config_for(
        tmp_path, {"method": "simgc", "dataset": "cora", "rate": "0.5", "seed": 1}
    )
    assert args.hidden == 1024 and args.eval_epochs == 3001 and args.activation is None


def test_linear_gcn_outputs_and_feature_gradients_match_author_operator(tmp_path):
    args = citation_args(tmp_path)
    model = GCN(2, 3, 2, args, mode="cross").double()
    model.eval()
    # The author's custom GCNConv propagates supplied normalized weights;
    # PyG GCNConv(normalize=False) performs the same operation.
    reference = torch.nn.ModuleList(
        [GCNConv(2, 3, normalize=False), GCNConv(3, 2, normalize=False)]
    ).double()
    for port, original in zip(model.layers, reference):
        with torch.no_grad():
            original.lin.weight.copy_(port.weight.T)
            original.bias.copy_(port.bias)
        assert torch.count_nonzero(port.bias) == 0
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    edge_weight = torch.tensor([0.8, 0.2, 0.2, 0.7], dtype=torch.double)
    adjacency = torch.tensor([[0.8, 0.2], [0.2, 0.7]], dtype=torch.double)
    x_port = torch.tensor([[1.0, -2.0], [0.5, 1.0]], dtype=torch.double, requires_grad=True)
    x_ref = x_port.detach().clone().requires_grad_()
    port = model(x_port, adjacency)
    original = reference[1](
        reference[0](x_ref, edge_index, edge_weight), edge_index, edge_weight
    ).log_softmax(dim=1)
    assert torch.allclose(port, original, atol=1e-12)
    port[:, 0].sum().backward()
    original[:, 0].sum().backward()
    assert torch.allclose(x_port.grad, x_ref.grad, atol=1e-12)


def test_validation_cadence_preserves_training_mode(tmp_path):
    args = citation_args(tmp_path)
    args.val_interval, args.hidden = 2, 4
    model = GCN(1, 4, 2, args, mode="cross")
    modes = []
    original = model.forward

    def observed(*a, **kw):
        modes.append(model.training)
        return original(*a, **kw)

    model.forward = observed
    model.fit_with_val(tiny_graph(), train_iters=5, reduced=True, setting="ind")
    assert modes == [True, False, True, True, False, True, True, False]


def test_public_evaluator_preserves_citation_zero_weight_decay(monkeypatch, tmp_path):
    args = citation_args(tmp_path)
    graph = tiny_graph()
    graph.labels_test = graph.labels_train
    graph.idx_test = graph.idx_val
    graph.adj_full = graph.adj_syn
    observed = {}

    def fake_fit(model, *a, **kw):
        observed.update(
            weight_decay=model.weight_decay,
            dropout=model.dropout,
            relu=model.with_relu,
            iters=kw["train_iters"],
        )
        return 0.5

    monkeypatch.setattr(GCN, "fit_with_val", fake_fit)
    Evaluator(args).test(graph, "GCN", verbose=False)
    assert observed == {"weight_decay": 0, "dropout": 0, "relu": False, "iters": 3001}


def test_other_methods_keep_default_relu_and_legacy_evaluation(tmp_path):
    args = get_args(["-M", "clustering", "-G", "-1", "--save_path", str(tmp_path)])
    model = GCN(1, 4, 2, copy(args), mode="eval")
    assert model.with_relu and model.dropout == 0 and model.weight_decay == 0.0005
    assert args.eval_epochs == 300

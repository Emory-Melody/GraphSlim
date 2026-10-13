"""Real reduction and real downstream optimization, not mocked integrations."""

from copy import deepcopy
from types import SimpleNamespace
import logging

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from graphslim.config import get_args
from graphslim.dataset import load_reduced
from graphslim.dataset.noise import NOISE_TYPES, cached_noise, graph_fingerprint, perturb_graph
from graphslim.evaluation import Evaluator
from graphslim.models import DOWNSTREAM_MODELS
from graphslim.reduction import create_reducer, get_method_spec
from graphslim.utils import seed_everything

METHODS = ("cgc", "clustgdd", "gcgp", "nope")


@pytest.mark.parametrize("method", METHODS)
def test_gat_keeps_small_constructed_edges(tmp_path, method):
    from graphslim.dataset.utils import sparsify

    args = quick_args(tmp_path, method, "trans")
    adjacency = torch.tensor([[0.01, 0.001], [0.001, 0.01]])
    preserved = sparsify("GAT", adjacency.clone(), args)
    assert torch.equal(preserved.to_dense(), adjacency)


def test_label_noise_invalidates_class_sampling_cache():
    graph = tiny_graph()
    graph.class_dict = {0: [0, 2]}
    graph.class_dict2 = {1: [1, 3]}
    graph.samplers = [object()]
    noisy = perturb_graph(graph, "label_noise", 0.5, 4)
    assert noisy.class_dict is noisy.class_dict2 is noisy.samplers is None
    assert graph.class_dict == {0: [0, 2]}


def test_noise_fingerprint_covers_heldout_split():
    first, second = tiny_graph(), tiny_graph()
    second.idx_val, second.idx_test = second.idx_test, second.idx_val
    assert graph_fingerprint(first, "trans") != graph_fingerprint(second, "trans")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_graph():
    generator = torch.Generator().manual_seed(91)
    n = 32
    labels = torch.arange(n) % 2
    features = torch.rand(n, 8, generator=generator) + labels[:, None].float()
    edges = [(i, (i + step) % n) for i in range(n) for step in (1, 2, 4)]
    adjacency = sp.csr_matrix((np.ones(len(edges)), np.array(edges).T), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T).astype(np.float32)
    graph = SimpleNamespace(
        feat_full=features,
        labels_full=labels,
        adj_full=adjacency,
        idx_train=torch.arange(16),
        idx_val=torch.arange(16, 24),
        idx_test=torch.arange(24, 32),
        nclass=2,
        num_nodes=n,
        x=features,
        y=labels,
        edge_index=torch.tensor(np.stack(adjacency.nonzero())),
    )
    for split in ("train", "val", "test"):
        index = getattr(graph, f"idx_{split}")
        setattr(graph, f"feat_{split}", features[index])
        setattr(graph, f"labels_{split}", labels[index])
        setattr(graph, f"adj_{split}", adjacency[index.numpy()][:, index.numpy()])
        mask = torch.zeros(n, dtype=torch.bool)
        mask[index] = True
        setattr(graph, f"{split}_mask", mask)
    return graph


def quick_args(tmp_path, method, setting):
    args = get_args(
        [
            "-M",
            method,
            "-G",
            "-1",
            "-R",
            "0.5",
            "--epochs",
            "2",
            "--eval_epochs",
            "2",
            "--run_eval",
            "1",
            "--save_path",
            str(tmp_path),
        ]
    )
    args.setting = setting
    args.hidden = 8
    args.clustgdd_hidden = 8
    args.clustgdd_pre_epochs = 2
    args.clustgdd_prop_steps = 2
    args.clustgdd_post_steps = 2
    args.cgc_clustering = "spectral"
    args.trans_num_layers = 1
    args.eval_mode = "cross"
    args.logger = logging.getLogger("lightweight-test")
    return args


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("noise", NOISE_TYPES)
def test_every_model_on_clean_and_training_noise(tmp_path, method, setting, noise):
    """4 methods x 2 settings x 7 conditions x 7 actual GNN fits = 392 fits."""
    args = quick_args(tmp_path, method, setting)
    source = tiny_graph()
    original_features = source.feat_full.clone()
    graph = perturb_graph(source, noise, 0.25, 17, setting=setting)
    seed_everything(args.seed)
    reducer = create_reducer(method, setting, graph, args)
    reduced = reducer.reduce(graph, verbose=False)
    assert reduced.feat_syn.shape[1] == source.feat_full.shape[1]
    assert reduced.adj_syn.shape == (len(reduced.feat_syn),) * 2
    assert torch.isfinite(reduced.feat_syn).all()
    assert not reduced.reduction_metadata["uses_test_labels"]
    assert torch.equal(source.feat_full, original_features)
    adjacency, features, labels = load_reduced(args)
    assert torch.equal(features, reduced.feat_syn)
    assert torch.equal(adjacency, reduced.adj_syn)
    assert torch.equal(labels, reduced.labels_syn)
    saved_adjacency = reduced.adj_syn.clone()
    for model in DOWNSTREAM_MODELS:
        evaluator = Evaluator(args)
        score, deviation = evaluator.evaluate(deepcopy(reduced), model, verbose=False)
        assert np.isfinite(score) and np.isfinite(deviation), (method, setting, noise, model)
        assert 0 <= score <= 1
    assert torch.equal(reduced.adj_syn, saved_adjacency)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_no_heldout_label_leakage_and_repeatability(tmp_path, method, setting):
    args = quick_args(tmp_path, method, setting)
    first, second = tiny_graph(), tiny_graph()
    second.labels_full = second.labels_full.clone()
    second.labels_full[second.idx_test] = 1 - second.labels_full[second.idx_test]
    second.labels_test = second.labels_full[second.idx_test]
    second.y = second.labels_full
    seed_everything(7)
    a = create_reducer(method, setting, first, args).reduce(first, save=False, verbose=False)
    seed_everything(7)
    b = create_reducer(method, setting, second, args).reduce(second, save=False, verbose=False)
    assert torch.equal(a.labels_syn, b.labels_syn)
    assert torch.allclose(a.feat_syn, b.feat_syn)
    assert torch.equal(a.adj_syn, b.adj_syn)


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("kind", NOISE_TYPES[1:])
def test_noise_seed_split_safety_and_cache(tmp_path, setting, kind):
    graph = tiny_graph()
    a = perturb_graph(graph, kind, 0.5, 9, setting=setting)
    b = perturb_graph(graph, kind, 0.5, 9, setting=setting)
    assert torch.equal(a.feat_full, b.feat_full)
    assert (a.adj_full != b.adj_full).nnz == 0
    assert torch.equal(a.labels_test, graph.labels_test)
    assert torch.equal(a.labels_val, graph.labels_val)
    if setting == "ind":
        assert torch.equal(a.feat_test, graph.feat_test)
        assert (a.adj_test != graph.adj_test).nnz == 0
    args = SimpleNamespace(
        attack=kind, ptb_r=0.5, seed=9, setting=setting, dataset="toy", save_path=str(tmp_path)
    )
    fresh = cached_noise(graph, args)
    restored = cached_noise(graph, args)
    assert torch.equal(fresh.feat_full, restored.feat_full)
    assert torch.equal(fresh.labels_train, restored.labels_train)
    assert (fresh.adj_train != restored.adj_train).nnz == 0
    assert (fresh.adj_full != restored.adj_full).nnz == 0
    assert torch.equal(cached_noise(fresh, args).feat_full, fresh.feat_full)
    alternate = tiny_graph()
    alternate.feat_full[0, 0] += 1
    alternate.feat_train[0, 0] += 1
    cached_noise(alternate, args)
    assert len(list(tmp_path.rglob("*.json"))) == 2


@pytest.mark.parametrize("method", METHODS)
def test_tiny_degenerate_and_missing_training_class(tmp_path, method):
    args = quick_args(tmp_path, method, "trans")
    graph = tiny_graph()
    graph.feat_full[:] = 0
    graph.feat_train[:] = 0
    graph.labels_train[:] = 0
    args.reduction_rate = 0.01
    result = create_reducer(method, "trans", graph, args).reduce(graph, save=False, verbose=False)
    assert len(result.feat_syn) >= 1
    assert torch.isfinite(result.feat_syn).all()


def test_public_classes_and_profiles(tmp_path):
    from graphslim.condensation import CGC, ClustGDD, GCGP
    from graphslim.coarsening import NOPE

    for name, cls in zip(METHODS, (CGC, ClustGDD, GCGP, NOPE)):
        assert get_method_spec(name).load_class() is cls
    args = get_args(
        ["-M", "ClustGDD", "-D", "cora", "-R", "0.25", "-G", "-1", "--save_path", str(tmp_path)]
    )
    assert args.epochs == 2000 and args.clustgdd_sampling_ratio == 0.06


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_inference_noise_nas_properties_and_mia(tmp_path, method, setting):
    from graphslim.evaluation.graph_property import PropertyEvaluator
    from graphslim.evaluation.nas_eval import NasEvaluator

    args = quick_args(tmp_path, method, setting)
    graph = tiny_graph()
    seed_everything(2)
    graph = create_reducer(method, setting, graph, args).reduce(graph, verbose=False)
    properties = PropertyEvaluator(args).evaluate(graph)
    assert properties["nodes"] == len(graph.feat_syn)
    assert all(value is None or np.isfinite(value) for value in properties.values())
    for model in DOWNSTREAM_MODELS:
        evaluator = Evaluator(args)
        _, score, mia = evaluator.test(graph, model, verbose=False, MIA=True, mode="cross")
        assert 0 <= score <= 1 and 0.5 <= mia <= 1
        weights = {key: value.clone() for key, value in evaluator.last_model.state_dict().items()}
        for noise in NOISE_TYPES[1:-1]:
            corrupted = perturb_graph(graph, noise, 0.25, 22, setting=setting, stage="inference")
            prefix = "full" if setting == "trans" else "test"
            prediction = evaluator.last_model.predict(
                getattr(corrupted, f"feat_{prefix}"), getattr(corrupted, f"adj_{prefix}")
            )
            assert torch.isfinite(prediction).all(), (method, setting, model, noise)
            assert all(
                torch.equal(value, weights[key])
                for key, value in evaluator.last_model.state_dict().items()
            )
    args.nas_epochs = 2
    nas = NasEvaluator(args)
    nas.parameter_combinations = [(2, 8, 0.1, "relu"), (2, 8, 0.2, "linear")]
    nas.evaluate_ori(graph)
    nas.evaluate_syn(graph)
    assert len(nas.results_ori) == len(nas.results_syn) == 2
    assert nas.best_params_ori is not None and nas.best_params_syn is not None


def test_gcgp_structure_kernel_really_updates_adjacency(tmp_path):
    args = quick_args(tmp_path, "gcgp", "trans")
    args.gcgp_kernel, args.gcgp_learn_adj = "SNTK", True
    graph = tiny_graph()
    result = create_reducer("gcgp", "trans", graph, args).reduce(graph, save=False, verbose=False)
    assert torch.equal(result.adj_syn, result.adj_syn.T)
    assert torch.equal(result.adj_syn.diagonal(), torch.ones(len(result.adj_syn)))
    assert torch.isfinite(result.feat_syn).all()
    args.gcgp_kernel = "SGNK"
    with pytest.raises(ValueError, match="requires SNTK"):
        create_reducer("gcgp", "trans", graph, args).reduce(graph, save=False)


def test_nope_isolates_are_not_silently_merged(tmp_path):
    args = quick_args(tmp_path, "nope", "trans")
    graph = tiny_graph()
    graph.adj_full = sp.csr_matrix(graph.adj_full.shape)
    with pytest.warns(RuntimeWarning, match="disconnected"):
        result = create_reducer("nope", "trans", graph, args).reduce(
            graph, save=False, verbose=False
        )
    assert not result.reduction_metadata["budget_reached"]
    assert len(result.feat_syn) == len(graph.idx_train)
    assert result.reduction_metadata["unlabeled_supernodes_removed"] == 16


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_nope_whole_graph_budget_is_explicit(tmp_path, setting):
    args = quick_args(tmp_path, "nope", setting)
    args.nope_budget_base = "graph"
    graph = tiny_graph()
    nodes = len(graph.feat_full if setting == "trans" else graph.feat_train)
    result = create_reducer("nope", setting, graph, args).reduce(graph, save=False, verbose=False)
    assert result.reduction_metadata["budget_base"] == "graph"
    assert result.reduction_metadata["requested_nodes"] == round(nodes * args.reduction_rate)
    assert len(result.nope_mapping) == nodes


def test_cross_without_search_uses_every_actual_model(tmp_path):
    args = quick_args(tmp_path, "cgc", "trans")
    graph = tiny_graph()
    graph = create_reducer("cgc", "trans", graph, args).reduce(graph, verbose=False)
    result = Evaluator(args).train_cross(graph, grid_search=False)
    assert tuple(result) == DOWNSTREAM_MODELS


def test_grid_search_does_not_unseal_test_before_selection(tmp_path, monkeypatch):
    args = quick_args(tmp_path, "cgc", "trans")
    evaluator = Evaluator(args)
    calls = []

    def fake_test(data, model_type, *, validation_only=False, **kwargs):
        calls.append((args.hidden, validation_only))
        return args.hidden / 100 if validation_only else (args.hidden / 100, 0.6)

    monkeypatch.setattr(evaluator, "test", fake_test)
    result, parameters = evaluator.grid_search(tiny_graph(), "GCN", {"hidden": [8, 16]})
    assert calls == [(8, True), (16, True), (16, False)]
    assert parameters == {"hidden": 16} and result == (0.6, 0.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_legacy_metattack_prbcd_and_all_models(tmp_path, method, setting):
    from graphslim.dataset.attack import attack

    args = quick_args(tmp_path, method, setting)
    args.attack, args.ptb_r = "metattack", 0.1
    args.attack_epochs = args.attack_pretrain_epochs = 2
    args.attack_search_space, args.attack_final_samples = 128, 2
    graph = attack(tiny_graph(), args)
    cached = attack(tiny_graph(), args)
    assert (graph.adj_train != cached.adj_train).nnz == 0
    assert (graph.adj_full != cached.adj_full).nnz == 0
    graph = create_reducer(method, setting, graph, args).reduce(graph, verbose=False)
    args.attack = None  # corruption is already frozen; no repeated attack
    for model in DOWNSTREAM_MODELS:
        score, deviation = Evaluator(args).evaluate(deepcopy(graph), model, verbose=False)
        assert np.isfinite(score) and np.isfinite(deviation)


@pytest.mark.parametrize("method", ["cgc", "nope", "gcgp"])
def test_optional_variants_support_all_downstream(tmp_path, method):
    args = quick_args(tmp_path, method, "trans")
    if method == "cgc":
        args.with_structure = 0
    elif method == "nope":
        args.nope_relaxed = True
    else:
        args.gcgp_kernel, args.gcgp_learn_adj = "SNTK", True
    graph = tiny_graph()
    graph = create_reducer(method, "trans", graph, args).reduce(graph, verbose=False)
    for model in DOWNSTREAM_MODELS:
        score, deviation = Evaluator(args).evaluate(deepcopy(graph), model, verbose=False)
        assert np.isfinite(score) and np.isfinite(deviation)

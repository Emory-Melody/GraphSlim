# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Graph corruption, noise caches, and adversarial cache handling on a tiny graph."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import importlib
from graphslim.dataset import noise

attack_module = importlib.import_module("graphslim.dataset.attack")
from test_lightweight_methods import tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def noise_args(tmp_path, kind, rate=0.25, setting="trans", seed=0):
    return SimpleNamespace(
        attack=kind,
        ptb_r=rate,
        seed=seed,
        setting=setting,
        save_path=str(tmp_path),
        dataset="tiny",
        device="cpu",
        logger=logging.getLogger("noise-test"),
    )


def undirected_edges(adjacency):
    row, col = sp.triu(sp.csr_matrix(adjacency), k=1).nonzero()
    return set(zip(row.tolist(), col.tolist()))


def test_fingerprint_depends_on_features_structure_and_splits():
    graph = tiny_graph()
    base = noise.graph_fingerprint(graph, "trans")
    assert base == noise.graph_fingerprint(tiny_graph(), "trans")
    assert noise.graph_fingerprint(graph, "ind") != base
    changed = tiny_graph()
    changed.idx_test = changed.idx_test.flip(0)
    assert noise.graph_fingerprint(changed, "trans") != base
    changed = tiny_graph()
    changed.feat_full = changed.feat_full + 1
    assert noise.graph_fingerprint(changed, "trans") != base


@pytest.mark.parametrize(
    "kind,rate,setting,stage,message",
    [
        ("unknown", 0.1, "trans", "training", "Unknown noise"),
        ("gaussian", 1.5, "trans", "training", "Unknown noise"),
        ("gaussian", 0.1, "both", "training", "Expected trans/ind"),
        ("gaussian", 0.1, "trans", "later", "Expected trans/ind"),
        ("label_noise", 0.1, "ind", "inference", "must not change ground-truth"),
    ],
)
def test_invalid_noise_requests_are_rejected(kind, rate, setting, stage, message):
    with pytest.raises(ValueError, match=message):
        noise.perturb_graph(tiny_graph(), kind, rate, setting=setting, stage=stage)


def test_label_noise_requires_two_classes():
    graph = tiny_graph()
    graph.nclass = 1
    with pytest.raises(ValueError, match="at least two classes"):
        noise.perturb_graph(graph, "label_noise", 0.5)


def test_edge_drop_removes_exact_fraction_and_refreshes_split_views():
    graph = tiny_graph()
    graph.sparse_adj = object()
    clean = undirected_edges(graph.adj_full)
    result = noise.perturb_graph(graph, "edge_drop", 0.25, seed=1)
    kept = undirected_edges(result.adj_full)
    assert kept < clean and len(clean) - len(kept) == int(0.25 * len(clean))
    assert (result.adj_full != result.adj_full.T).nnz == 0
    index = graph.idx_train.numpy()
    assert (result.adj_train != result.adj_full[index][:, index]).nnz == 0
    assert result.sparse_adj is None
    assert result.edge_index.shape[1] == result.adj_full.nnz
    assert (graph.adj_full != tiny_graph().adj_full).nnz == 0  # input untouched


@pytest.mark.parametrize("n,rate", [(32, 0.5), (6, 1.0)])
def test_random_edges_are_new_symmetric_and_bounded_by_capacity(n, rate):
    graph = tiny_graph()
    if n != 32:
        # A nearly complete graph forces the exhaustive candidate path.
        dense = np.ones((n, n)) - np.eye(n)
        dense[0, 1] = dense[1, 0] = dense[2, 3] = dense[3, 2] = 0
        graph.adj_train = sp.csr_matrix(dense)
    prefix = "full" if n == 32 else "train"
    setting = "trans" if n == 32 else "ind"
    before = undirected_edges(getattr(graph, f"adj_{prefix}"))
    result = noise.perturb_graph(graph, "random_adj", rate, seed=2, setting=setting)
    after = undirected_edges(getattr(result, f"adj_{prefix}"))
    capacity = n * (n - 1) // 2 - len(before)
    assert before < after
    assert len(after - before) == min(int(rate * len(before)), capacity)
    if setting == "ind":
        assert (result.adj_full != graph.adj_full).nnz == 0


@pytest.mark.parametrize("kind", ["random_feat", "feature_mask", "gaussian"])
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_feature_noise_changes_only_the_visible_split(kind, setting):
    graph = tiny_graph()
    result = noise.perturb_graph(graph, kind, 0.5, seed=4, setting=setting)
    if setting == "trans":
        assert not torch.equal(result.feat_full, graph.feat_full)
        torch.testing.assert_close(result.feat_test, result.feat_full[graph.idx_test])
    else:
        assert not torch.equal(result.feat_train, graph.feat_train)
        torch.testing.assert_close(result.feat_full[graph.idx_train], result.feat_train)
        torch.testing.assert_close(result.feat_test, graph.feat_test)
    assert result.x is result.feat_full
    if kind == "random_feat":
        zeroed = (result.feat_train == 0).all(0)
        assert int(zeroed.sum()) == 4
    assert result.noise_metadata == {
        "kind": kind,
        "rate": 0.5,
        "seed": 4,
        "setting": setting,
        "stage": "training",
    }


def test_inference_noise_targets_test_features_only():
    graph = tiny_graph()
    result = noise.perturb_graph(graph, "gaussian", 1.0, seed=0, setting="ind", stage="inference")
    torch.testing.assert_close(result.feat_train, graph.feat_train)
    assert not torch.equal(result.feat_test, graph.feat_test)
    torch.testing.assert_close(result.feat_full[graph.idx_test], result.feat_test)


def test_label_noise_flips_exact_count_to_other_classes():
    graph = tiny_graph()
    graph.class_dict = {"stale": True}
    result = noise.perturb_graph(graph, "label_noise", 0.5, seed=3)
    changed = result.labels_train != graph.labels_train
    assert int(changed.sum()) == 8
    torch.testing.assert_close(result.labels_full[graph.idx_train], result.labels_train)
    torch.testing.assert_close(result.labels_full[graph.idx_test], graph.labels_test)
    assert result.class_dict is None and result.y is result.labels_full


def test_zero_rate_and_clean_noise_only_record_metadata():
    graph = tiny_graph()
    for kind in ("clean", "gaussian"):
        result = noise.perturb_graph(graph, kind, 0.0)
        torch.testing.assert_close(result.feat_full, graph.feat_full)
        assert (result.adj_full != graph.adj_full).nnz == 0
        assert result.noise_metadata["kind"] == kind


@pytest.mark.parametrize("setting", ["trans", "ind"])
@pytest.mark.parametrize("kind", ["random_adj", "gaussian", "label_noise"])
def test_cached_noise_roundtrip_reproduces_generated_graph(tmp_path, setting, kind):
    graph = tiny_graph()
    args = noise_args(tmp_path, kind, setting=setting)
    generated = noise.cached_noise(graph, args)
    files = sorted(p.suffix for p in (tmp_path / "corrupt_graph" / kind).iterdir())
    assert files == [".json", ".pt"]
    assert noise.cached_noise(generated, args) is generated
    fresh = tiny_graph()
    fresh.samplers = ["stale"]
    loaded = noise.cached_noise(fresh, args)
    prefix = "full" if setting == "trans" else "train"
    torch.testing.assert_close(
        getattr(loaded, f"feat_{prefix}"), getattr(generated, f"feat_{prefix}")
    )
    assert (getattr(loaded, f"adj_{prefix}") != getattr(generated, f"adj_{prefix}")).nnz == 0
    torch.testing.assert_close(loaded.labels_train, generated.labels_train)
    torch.testing.assert_close(loaded.labels_full[graph.idx_test], graph.labels_test)
    assert loaded.samplers is None and loaded.noise_metadata == generated.noise_metadata
    if setting == "trans":
        torch.testing.assert_close(loaded.feat_val, loaded.feat_full[graph.idx_val])
    else:
        torch.testing.assert_close(loaded.feat_full[graph.idx_train], loaded.feat_train)


@pytest.mark.parametrize(
    "corrupt,message",
    [
        (lambda v: v.pop("labels"), "Invalid noise cache fields"),
        (lambda v: v.update(features=torch.zeros(3, 3)), "feature shape mismatch"),
        (lambda v: v.update(shape=(5, 5)), "adjacency shape mismatch"),
    ],
)
def test_corrupt_noise_cache_is_rejected(tmp_path, corrupt, message):
    args = noise_args(tmp_path, "gaussian")
    noise.cached_noise(tiny_graph(), args)
    payload = next((tmp_path / "corrupt_graph" / "gaussian").glob("*.pt"))
    values = torch.load(payload, weights_only=True)
    corrupt(values)
    torch.save(values, payload)
    with pytest.raises(ValueError, match=message):
        noise.cached_noise(tiny_graph(), args)


def test_attack_dispatch_validation_and_noise_delegation(tmp_path):
    graph = tiny_graph()
    with pytest.raises(ValueError, match="Unknown attack"):
        attack_module.attack(graph, noise_args(tmp_path, "nettack"))
    with pytest.raises(ValueError, match="ptb_r"):
        attack_module.attack(graph, noise_args(tmp_path, "metattack", rate=1.5))
    delegated = attack_module.attack(graph, noise_args(tmp_path, "edge_drop", rate=0.5))
    assert delegated.noise_metadata["kind"] == "edge_drop"
    assert len(undirected_edges(delegated.adj_full)) < len(undirected_edges(graph.adj_full))


def test_metattack_with_empty_budget_keeps_graph_and_short_circuits(tmp_path):
    graph = tiny_graph()
    graph.sparse_adj = object()
    args = noise_args(tmp_path, "metattack", rate=0.001)
    result = attack_module.attack(graph, args)
    assert (result.adj_full != graph.adj_full).nnz == 0
    assert result.sparse_adj is None
    assert result.edge_index.shape[1] == graph.adj_full.nnz
    assert attack_module.attack(result, args) is result


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_metattack_runs_prbcd_once_and_reloads_its_cache(tmp_path, setting):
    graph = tiny_graph()
    args = noise_args(tmp_path, "metattack", rate=0.1, setting=setting)
    args.attack_epochs = 2
    args.attack_pretrain_epochs = 2
    args.attack_search_space = 64
    args.attack_final_samples = 2
    prefix = "full" if setting == "trans" else "train"
    first = attack_module.attack(graph, args)
    adjacency = getattr(first, f"adj_{prefix}")
    assert adjacency.shape == getattr(graph, f"adj_{prefix}").shape
    assert (adjacency != getattr(graph, f"adj_{prefix}")).nnz > 0
    folder = tmp_path / "corrupt_graph" / "metattack"
    metadata = json.loads(next(folder.glob("*.json")).read_text())
    assert metadata["implementation"].startswith("PRBCD")
    assert metadata["budget"]["epochs"] == 2
    second = attack_module.attack(tiny_graph(), args)
    assert (getattr(second, f"adj_{prefix}") != adjacency).nnz == 0
    sp.save_npz(next(folder.glob("*.npz")), sp.csr_matrix((3, 3)))
    with pytest.raises(ValueError, match="cache shape mismatch"):
        attack_module.attack(tiny_graph(), args)

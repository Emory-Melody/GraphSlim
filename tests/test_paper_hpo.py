from types import SimpleNamespace

import pytest
import torch

from benchmark import paper_hpo as hpo
from benchmark import rerun_nr as nr
from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.simgc import SimGCTeacher, propagated_features, smoothness_loss
from graphslim.config import get_args
from graphslim.models import GCN


def tiny_graph():
    features = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([0, 1])
    return SimpleNamespace(
        nclass=2,
        feat_full=features,
        feat_syn=features,
        feat_train=features,
        feat_val=features,
        labels_train=labels,
        labels_val=labels,
        labels_syn=labels,
        adj_syn=torch.eye(2),
        adj_train=torch.eye(2),
        adj_val=torch.eye(2),
        idx_val=torch.arange(2),
    )


def test_paper_search_grids_and_roster_are_exact():
    assert len(hpo.gcn_grid()) == 108
    assert len(hpo.gecc_grid()) * 2 == 4394
    assert {row["agg_gamma"] for row in hpo.gecc_grid()} == {
        round(i / 10, 1) for i in range(-3, 10)
    }
    jobs = hpo.roster()
    assert len(jobs) == len({job["id"] for job in jobs}) == 369
    assert sum(job["kind"] == "reduce" for job in jobs) == 63
    assert sum(job["kind"] == "final" for job in jobs) == 72
    assert sum(len(job["candidates"]) for job in jobs if job["kind"] == "gcn") == 758
    assert sum(len(job["candidates"]) for job in jobs if job["kind"] == "gecc") == 2197
    assert all(
        dependency in {job["id"] for job in jobs} for job in jobs for dependency in job["depends"]
    )


def test_simgc_multihop_alignment_has_feature_and_edge_gradients():
    features = torch.tensor([[1.0], [3.0]], requires_grad=True)
    adjacency = torch.tensor([[0.7, 0.3], [0.2, 0.8]], requires_grad=True)
    multi = propagated_features(features, adjacency, 2)
    multi[:, 1:].square().sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert adjacency.grad is not None and adjacency.grad.abs().sum() > 0


def test_simgc_smoothness_matches_released_negative_similarity_and_empty_edges():
    features = torch.tensor([[1.0], [1.0]], requires_grad=True)
    assert smoothness_loss(features, torch.tensor([[0], [1]]), torch.ones(1)).item() == -1
    loss = smoothness_loss(features, torch.empty((2, 0), dtype=torch.long), torch.empty(0))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()


def test_simgc_teacher_predicts_after_propagation(tmp_path):
    args = get_args(["-D", "cora", "-M", "simgc", "-G", "-1", "--save_path", str(tmp_path)])
    args.ntrans, args.nlayers, args.with_bn, args.dropout = 2, 2, False, 0
    teacher = SimGCTeacher(1, 3, 2, args)
    features = torch.tensor([[1.0], [-2.0]])
    adjacency = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    hidden = torch.relu(teacher.layers[0](adjacency @ (adjacency @ features)))
    expected = torch.log_softmax(teacher.layers[1](hidden), dim=1)
    assert torch.allclose(teacher(features, adjacency), expected)


def test_simgc_teacher_regularization_cannot_inherit_gcn_grid(monkeypatch, tmp_path):
    import scipy.sparse as sp
    from graphslim.condensation.simgc import SimGC

    args = get_args(["-D", "cora", "-M", "simgc", "-G", "-1", "--save_path", str(tmp_path)])
    args.weight_decay, args.teacher_hidden = 0.001, 4
    args.epochs, args.checkpoints, args.validation_only = 0, [], True
    features = torch.arange(1, 5, dtype=torch.float32).view(-1, 1)
    graph = SimpleNamespace(
        feat_full=features,
        feat_train=features,
        adj_full=sp.eye(4, format="csr"),
        labels_train=torch.tensor([0, 0, 1, 1]),
        nclass=2,
        idx_train=torch.arange(4),
    )
    observed = {}

    def fake_fit(teacher, *a, **kw):
        observed.update(weight_decay=teacher.weight_decay, lr=teacher.lr, normalize=kw["normadj"])
        return 0.5

    monkeypatch.setattr(SimGCTeacher, "fit_with_val", fake_fit)
    SimGC("trans", graph, args).reduce(graph, verbose=False)
    assert observed == {"weight_decay": 0, "lr": 0.001, "normalize": True}
    assert args.weight_decay == 0.001


def test_gecc_kmeanspp_handles_no_old_centers_and_empty_request():
    from graphslim.condensation.gecc import incremental_kmeanspp_init

    points = torch.arange(6, dtype=torch.float32).view(-1, 1)
    centers = incremental_kmeanspp_init(points, None, 4, device="cpu")
    assert len(torch.unique(centers)) == 4
    assert incremental_kmeanspp_init(points, None, 0, device="cpu").shape == (0, 1)


def test_validation_only_condensation_never_calls_test(monkeypatch, tmp_path):
    args = get_args(["-D", "cora", "-M", "simgc", "-G", "-1", "--save_path", str(tmp_path)])
    args.validation_only, args.eval_mode = True, "cross"
    reducer = GCondBase.__new__(GCondBase)
    reducer.args, reducer.data, reducer.device = args, tiny_graph(), "cpu"

    def forbidden(*a, **kw):
        pytest.fail("Test metric accessed during validation-only condensation")

    monkeypatch.setattr(GCN, "test", forbidden)
    assert len(reducer.test_with_val(setting="ind", iters=2)) == 1


def test_gcn_hpo_preserves_parameters_and_restores_training_mode(tmp_path):
    args = get_args(["-D", "cora", "-M", "clustering", "-G", "-1", "--save_path", str(tmp_path)])
    args.dropout, args.weight_decay = 0.5, 0.001
    model = GCN(1, 4, 2, args, mode="cross")
    assert model.dropout == 0.5 and model.weight_decay == 0.001
    modes = []
    original = model.forward

    def observed(*a, **kw):
        modes.append(model.training)
        return original(*a, **kw)

    model.forward = observed
    model.fit_with_val(tiny_graph(), train_iters=3, reduced=True, setting="ind")
    assert modes == [True, False] * 3


def test_hpo_selection_requires_full_coverage_and_never_uses_test(tmp_path):
    jobs = [{"id": "search1", "kind": "gcn", "method": "simgc", "dataset": "cora"}]
    manifest = {"source_sha256": "frozen", "jobs": jobs}
    assert not (tmp_path / "selected").exists()
    hpo.select_complete_groups(tmp_path, manifest, {"search1": "pending"})
    candidates = [
        hpo.candidate_record(
            i, {"hidden": 64}, [{"seed": seed, "validation": 0.5} for seed in range(10)]
        )
        for i in range(108)
    ]
    candidates[3]["validation_mean"] = 0.8
    candidates[9]["validation_mean"] = 0.8
    for index in (3, 9):
        candidates[index]["evaluations"] = [{"seed": seed, "validation": 0.8} for seed in range(10)]
    nr.write_json(
        tmp_path / "jobs/search1/result.json",
        {"source_sha256": "frozen", "data": {"fixed": True}, "candidates": candidates},
    )
    hpo.select_complete_groups(tmp_path, manifest, {"search1": "complete"})
    selected = nr.read_json(tmp_path / "selected/simgc__cora.json")
    assert selected["winner"]["id"] == 3
    assert selected["candidate_count"] == 108
    assert "test" not in selected["winner"]["evaluations"][0]

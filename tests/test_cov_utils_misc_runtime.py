"""Coverage for CLI entry points, tracking, visualization, doctor, config, compat and HPO."""

import json
import sys
from types import SimpleNamespace

import click
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from click.testing import CliRunner
from torch_geometric.data import Data

from graphslim import compat, config, doctor, hpo, run_eval, tracking, train_all, visualization
from graphslim._lazy import lazy_exports
from test_lightweight_methods import quick_args, tiny_graph


# --------------------------------------------------------------------------- _lazy / layers


def test_lazy_exports_resolve_cache_and_list_names():
    namespace = {"__name__": "graphslim", "existing": 1}
    names, get_attribute, list_attributes = lazy_exports(
        namespace, {"dumps": ("json", "dumps"), "Path": ("pathlib", "Path")}
    )
    assert names == ["dumps", "Path"]
    assert list_attributes() == sorted(["__name__", "existing", "dumps", "Path"])
    assert get_attribute("dumps") is json.dumps
    assert namespace["dumps"] is json.dumps
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        get_attribute("missing")


def test_layers_package_reexports_model_layers():
    import graphslim.layers as layers
    import graphslim.models.layers as model_layers

    assert set(layers.__all__) == {
        "ChebConvolution",
        "GATConv",
        "GraphConvolution",
        "MyLinear",
        "SageConvolution",
    }
    for name in layers.__all__:
        assert getattr(layers, name) is getattr(model_layers, name)


# --------------------------------------------------------------------------- train_all / run_eval


class FakeTracker:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step=None):
        self.calls.append(("log", dict(metrics), step))

    def log_graph(self, name, graph, step=None):
        self.calls.append(("graph", name, step))

    def finish(self):
        self.calls.append(("finish",))


class FakeGraph(SimpleNamespace):
    def reset(self):
        self.resets = getattr(self, "resets", 0) + 1


def _run_args(tmp_path, **overrides):
    args = SimpleNamespace(
        device="cpu",
        seed=5,
        dataset="cora",
        load_path=str(tmp_path),
        attack=None,
        hpo=False,
        eval_whole=False,
        run_reduction=3,
        method="random",
        setting="trans",
        verbose=False,
        eval_model="GCN",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_fakes(monkeypatch, tracker, reduced_seeds, evaluated):
    import graphslim.evaluation

    class Reducer:
        def __init__(self, args):
            self.args = args

        def reduce(self, graph, verbose):
            reduced_seeds.append(self.args.seed)
            return SimpleNamespace(seed=self.args.seed)

    def create(method, setting, data, args):
        assert (method, setting) == ("random", "trans")
        return Reducer(args)

    class Evaluator:
        def __init__(self, args):
            self.args = args

        def evaluate(self, graph, model_type, reduced):
            evaluated.append((graph, model_type, reduced, self.args.seed))
            return (float(self.args.seed), 0.25)

    monkeypatch.setattr(train_all, "create_reducer", create)
    monkeypatch.setattr(train_all, "build_tracker", lambda args: tracker)
    monkeypatch.setattr(graphslim.evaluation, "Evaluator", Evaluator, raising=False)


def test_run_repeats_reduction_with_offset_seeds_and_logs_each_result(tmp_path, monkeypatch):
    tracker, seeds, evaluated, hooked = FakeTracker(), [], [], []
    _install_fakes(monkeypatch, tracker, seeds, evaluated)
    graph = FakeGraph()
    args = _run_args(tmp_path)
    results = train_all.run(
        args,
        load_dataset=lambda name, a, path: graph if (name, path) == ("cora", str(tmp_path)) else 0,
        on_result=lambda run_args, reduced, evaluator, result: hooked.append(run_args.seed),
    )
    assert seeds == [5, 6, 7] and hooked == [5, 6, 7]
    assert results == [(5.0, 0.25), (6.0, 0.25), (7.0, 0.25)]
    assert graph.resets == 2
    assert all(model == "GCN" and reduced for _, model, reduced, _ in evaluated)
    assert tracker.calls[0] == ("graph", "original", 0)
    assert ("graph", "reduced", 3) in tracker.calls
    assert tracker.calls[-2] == (
        "log",
        {"eval/mean": 7.0, "eval/std": 0.25, "reduction/seed": 7},
        3,
    )
    assert tracker.calls[-1] == ("finish",)
    assert args.seed == 5


def test_run_whole_graph_skips_reduction_and_applies_attack(tmp_path, monkeypatch):
    import graphslim.dataset

    tracker, seeds, evaluated = FakeTracker(), [], []
    _install_fakes(monkeypatch, tracker, seeds, evaluated)
    attacked = FakeGraph(name="attacked")
    monkeypatch.setattr(graphslim.dataset, "attack", lambda graph, args: attacked, raising=False)
    args = _run_args(tmp_path, eval_whole=True, attack="random_adj")
    results = train_all.run(args, load_dataset=lambda *a: FakeGraph(name="clean"))
    assert results == [(5.0, 0.25)]
    assert seeds == [] and evaluated[0][0] is attacked and evaluated[0][2] is False


def test_run_finishes_tracker_when_evaluation_fails(tmp_path, monkeypatch):
    import graphslim.evaluation

    tracker = FakeTracker()
    _install_fakes(monkeypatch, tracker, [], [])

    class Broken:
        def __init__(self, args):
            pass

        def evaluate(self, *args, **kwargs):
            raise RuntimeError("evaluation failed")

    monkeypatch.setattr(graphslim.evaluation, "Evaluator", Broken, raising=False)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        train_all.run(_run_args(tmp_path, run_reduction=0), load_dataset=lambda *a: FakeGraph())
    assert tracker.calls[-1] == ("finish",)


def test_run_delegates_to_hpo_and_selects_cuda_device(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))
    import graphslim.utils

    monkeypatch.setattr(graphslim.utils, "seed_everything", lambda seed: calls.append(seed))
    monkeypatch.setattr(hpo, "run_search", lambda args, graph: ("best", graph))
    graph = FakeGraph()
    args = _run_args(tmp_path, device="cuda:1", hpo=True)
    assert train_all.run(args, load_dataset=lambda *a: graph) == ("best", graph)
    assert calls == ["cuda:1", 5]


def test_train_main_and_entrypoints_dispatch_parsed_arguments(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(train_all, "run", lambda args: seen.append(args) or "done")
    argv = ["-G", "-1", "--epochs", "1", "--save_path", str(tmp_path)]
    assert train_all.main(argv, default_method="random") == "done"
    assert seen[-1].method == "random" and seen[-1].device == "cpu"
    assert train_all.main(["--help"]) is None
    assert "Usage" in capsys.readouterr().out

    monkeypatch.setattr(run_eval, "run", lambda args: seen.append(args) or "evaluated")
    assert run_eval.main(argv + ["--run_reduction", "2"]) == "evaluated"
    assert seen[-1].run_reduction == 0
    assert run_eval.main(["--help"]) is None

    entry_calls = []
    monkeypatch.setattr(train_all, "main", lambda: entry_calls.append("train"))
    monkeypatch.setattr(run_eval, "main", lambda: entry_calls.append("eval") or 1)
    assert train_all.entrypoint() is None and run_eval.entrypoint() is None
    assert entry_calls == ["train", "eval"]


# --------------------------------------------------------------------------- tracking


def test_graph_summary_counts_nodes_edges_and_density():
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    assert tracking.graph_summary(SimpleNamespace(num_nodes=4, edge_index=edges)) == {
        "nodes": 4,
        "edges": 3,
        "density": 3 / 12,
    }
    assert tracking.graph_summary(SimpleNamespace(x=torch.ones(5, 2))) == {"nodes": 5}
    sparse = SimpleNamespace(feat_syn=torch.ones(3, 1), adj_syn=torch.eye(3).to_sparse())
    assert tracking.graph_summary(sparse) == {"nodes": 3, "edges": 3, "density": 0.5}
    scipy_graph = SimpleNamespace(adj_syn=sp.csr_matrix(np.ones((2, 2))))
    assert tracking.graph_summary(scipy_graph) == {"edges": 4}
    binary = SimpleNamespace(adj_syn=np.array([[0, 1], [1, 0]]))
    assert tracking.graph_summary(binary) == {"edges": 2}
    single = tracking.graph_summary(SimpleNamespace(num_nodes=1, edge_index=edges[:, :0]))
    assert single == {"nodes": 1, "edges": 0, "density": 0.0}


def test_null_tracker_is_inert():
    tracker = tracking.build_tracker(None)
    assert isinstance(tracker, tracking.NullTracker) and not tracker.enabled
    assert tracker.log({"a": 1}, step=1) is None
    assert tracker.log_graph("g", SimpleNamespace(), step=1) is None
    assert tracker.finish() is None


def test_wandb_tracker_uses_argument_defaults(monkeypatch):
    calls = []
    client = SimpleNamespace(
        init=lambda **kw: calls.append(kw) or "run",
        log=lambda metrics, step=None: calls.append((metrics, step)),
        finish=lambda: calls.append("finish"),
    )
    monkeypatch.setitem(sys.modules, "wandb", client)
    tracker = tracking.WandbTracker(SimpleNamespace(count=1, ratio=0.5, items=[1]))
    assert tracker.enabled and tracker._run == "run"
    assert calls[0] == {"project": "graphslim", "name": None, "config": {"count": 1, "ratio": 0.5}}
    explicit = tracking.WandbTracker(None, project="p", run_name="r")
    assert calls[1] == {"project": "p", "name": "r", "config": {}}
    explicit.log({"x": 1})
    assert calls[-1] == ({"x": 1}, None)


# --------------------------------------------------------------------------- visualization


def test_visualization_handles_missing_edges_and_samples_large_graphs(tmp_path):
    assert visualization.to_networkx_graph(Data()).number_of_nodes() == 0
    ring = Data(edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]), x=torch.ones(5, 1))
    fig, axes = visualization.draw_graph_pair(ring, ring, title="T", max_nodes=2)
    try:
        assert fig._suptitle.get_text() == "T"
        assert axes[0].get_title() == "Original: 5 nodes, 5 edges"
        assert axes[1].get_title() == "Reduced: 5 nodes, 5 edges"
        assert len(axes[0].collections[0].get_offsets()) == 2
    finally:
        plt.close(fig)


def test_reduced_graph_loader_accepts_array_like_adjacency(tmp_path, monkeypatch):
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    monkeypatch.setattr(visualization.torch, "load", lambda path, map_location: adjacency)
    data = visualization.load_reduced_graph("unused.pt")
    assert data.edge_index.tolist() == [[0, 1, 2], [1, 2, 0]]
    assert data.edge_index.dtype == torch.long


# --------------------------------------------------------------------------- doctor


def test_version_lookup_reports_missing_distribution():
    assert doctor.version("graphslim-surely-not-installed-xyz") is None
    assert doctor.version("numpy") == np.__version__


def test_doctor_reports_unsupported_python_and_prints_text(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "sys", SimpleNamespace(version_info=(3, 9, 0)))
    with pytest.raises(SystemExit, match="1"):
        doctor.main([])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("GraphSlim runtime probe: Python")
    assert any(line.startswith("python_support: failed") and "3.10/3.11" in line for line in lines)
    assert "pyg_gcn: passed " in lines


def test_doctor_text_output_succeeds_on_cpu(capsys):
    doctor.main([])
    assert "torch_native_sparse: passed" in capsys.readouterr().out


def test_doctor_requested_cuda_without_runtime_fails_fast(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = doctor.diagnose("cuda:0")
    assert report["checks"]["torch_import"]["status"] == "failed"
    assert "Torch cannot access CUDA" in report["checks"]["torch_import"]["error"]
    assert "torch_native_sparse" not in report["checks"]


def test_doctor_cuda_probe_records_device_and_synchronizes(monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synced.append(str(device)))
    report = doctor.diagnose("cuda:0")
    assert report["gpu"] == {
        "name": "Fake GPU",
        "capability": [9, 0],
        "torch_architectures": ["sm_90"],
    }
    assert report["checks"]["cuda_synchronize"]["status"] == "passed"
    assert synced == ["cuda:0"]


def test_doctor_dgl_roundtrip_passes_with_installed_backend():
    pytest.importorskip("dgl")
    report = doctor.diagnose(dgl=True)
    assert report["checks"]["dgl_roundtrip_cpu"]["status"] == "passed"


# --------------------------------------------------------------------------- config


def test_config_objects_tables_and_value_formatting():
    obj = config.dict2obj({"a": 1, "nested": {"b": [1, 2]}})
    assert obj.nested.b == [1, 2]
    assert "'a': 1" in repr(obj)
    assert config._build_table([]) == ""
    assert config.format_args_tables(SimpleNamespace()) == ""
    assert config._stringify_value(0.1234567) == "0.123457"
    assert config._stringify_value(len) == "len"
    assert config._stringify_value({"k": [1, 2]}) == "{'k': [1, 2]}"


def test_config_tables_filter_method_specific_and_attack_sections():
    args = SimpleNamespace(
        method="kcenter", dataset="cora", no_buff=True, attack="none", custom_key=3
    )
    text = config.format_args_tables(args)
    assert "no_buff" not in text and "KCENTER" in text and "custom_key" in text
    assert "Attack" not in text
    args.method, args.attack, args.ptb_r = "geom", "random_adj", 0.1
    text = config.format_args_tables(args)
    assert "no_buff" in text and "Attack" in text
    assert config._should_include_category("MSGC", SimpleNamespace(method=None)) is False
    assert config._should_include_category("MSGC", SimpleNamespace(method="MSGC")) is True
    unnamed = config.format_args_tables(SimpleNamespace(method=None, mystery=1))
    assert unnamed.startswith("Common") and "mystery" in unnamed


@pytest.mark.parametrize(
    ("dataset", "setting", "expected_setting", "rate"),
    [("flickr", "trans", "ind", 0.01), ("photo", None, "trans", 0.1), ("yelp", None, "ind", 0.001)],
)
def test_setting_config_assigns_dataset_defaults(dataset, setting, expected_setting, rate):
    args = SimpleNamespace(
        dataset=dataset, setting=setting, reduction_rate=-1, epochs=25, _method_config_keys=set()
    )
    config.setting_config(args)
    assert args.setting == expected_setting and args.reduction_rate == rate
    assert args.eval_interval == 2 and args.checkpoints[:3] == [-1, 1, 3]
    assert args.metric.__name__ == ("f1_macro" if dataset == "yelp" else "accuracy")


def test_cli_validates_json_options_and_edge_reduction(tmp_path):
    base = ["-G", "-1", "--epochs", "1", "--save_path", str(tmp_path)]
    args = config.get_args(base + ["--sparsifier_options", '{"maxRank": 3}'])
    assert args.sparsifier_options == {"maxRank": 3}
    runner = CliRunner()
    for bad in ["{oops", "[1, 2]"]:
        result = runner.invoke(config.cli, base + ["--sparsifier_options", bad])
        assert result.exit_code == 2 and "must be a JSON object" in result.output
    result = runner.invoke(config.cli, base + ["-M", "random", "--edge_reduction"])
    assert result.exit_code == 2 and "node reduction method" in result.output
    edge = config.get_args(base + ["-M", "t_spanner"])
    assert edge.edge_reduction is True


def test_cli_config_file_overrides_and_rejections(tmp_path):
    base = ["-G", "-1", "--epochs", "1", "--save_path", str(tmp_path)]
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"hidden": 7}))
    args = config.get_args(base + ["--config", str(good)])
    assert args.hidden == 7 and "hidden" in args._method_config_keys
    runner = CliRunner()
    for payload, message in [([1], "JSON object"), ({"dataset": "x"}, "Use CLI options")]:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        result = runner.invoke(config.cli, base + ["--config", str(bad)])
        assert result.exit_code == 2 and message in result.output


def test_method_specific_checkpoint_and_batch_rules(tmp_path):
    base = ["-G", "-1", "--save_path", str(tmp_path)]
    simgc = config.get_args(base + ["-M", "simgc", "--epochs", "7", "--eval_interval", "3"])
    assert simgc.checkpoints == [3, 6, 7]
    msgc = config.method_config(SimpleNamespace(method="msgc", dataset="no-such-dataset"))
    assert msgc.batch_adj == 16


def test_method_config_merges_list_valued_existing_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "__file__", str(tmp_path / "config.py"))
    target = tmp_path / "configs/gcond/cora.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"lr": 0.5}))
    args = SimpleNamespace(method="gcond", dataset="cora", _method_config_keys=["epochs"])
    config.method_config(args)
    assert args.lr == 0.5 and args._method_config_keys == {"epochs", "lr"}


# --------------------------------------------------------------------------- compat


class FakeDGLGraph:
    __module__ = "dgl.heterograph"

    def __init__(self, homogeneous=True):
        self.is_homogeneous = homogeneous
        self.ndata = {"feature": torch.ones(3, 2), "label": torch.tensor([0, 1, 0])}
        self.edata = {"edge_weight": torch.tensor([0.5, 2.0])}

    def edges(self):
        return torch.tensor([0, 1]), torch.tensor([1, 2])

    def num_nodes(self):
        return 3


def test_compat_reads_dgl_feature_alias_and_edge_weight_alias():
    assert compat.is_dgl_graph(FakeDGLGraph())
    assert compat.is_pyg_data(Data()) and not compat.is_pyg_data(FakeDGLGraph())
    data = compat.to_pyg_data(FakeDGLGraph())
    torch.testing.assert_close(data.x, torch.ones(3, 2))
    torch.testing.assert_close(data.edge_weight, torch.tensor([0.5, 2.0]))
    assert data.edge_index.tolist() == [[0, 1], [1, 2]] and data.num_nodes == 3
    assert not hasattr(data, "train_mask") or data.train_mask is None


def test_compat_rejects_unsupported_reduced_adjacency():
    with pytest.raises(ValueError, match="one reduced adjacency"):
        compat.to_pyg_data(SimpleNamespace(adj_syn=torch.ones(2, 2, 2)))
    with pytest.raises(TypeError, match="Unsupported reduced adjacency"):
        compat.to_pyg_data(SimpleNamespace(adj_syn=[[0, 1], [1, 0]]))


def test_compat_keeps_declared_node_count_without_features():
    graph = SimpleNamespace(
        edge_index=torch.tensor([[0], [1]]),
        num_nodes=5,
        train_mask=torch.ones(5, dtype=torch.bool),
    )
    data = compat.to_pyg_data(graph)
    assert data.num_nodes == 5 and data.train_mask.all()


def test_to_dgl_reports_missing_runtime_and_passes_through(monkeypatch):
    graph = FakeDGLGraph()
    monkeypatch.setitem(sys.modules, "dgl", SimpleNamespace(graph=None))
    assert compat.to_dgl_graph(graph) is graph
    with pytest.raises(ValueError, match="without edge_index"):
        compat.to_dgl_graph(Data(x=torch.ones(2, 1)))
    monkeypatch.setitem(sys.modules, "dgl", None)
    with pytest.raises(RuntimeError, match="DGL could not be imported"):
        compat.to_dgl_graph(Data(edge_index=torch.tensor([[0], [1]])))


def test_to_dgl_copies_edge_attributes_to_fake_backend(monkeypatch):
    class Graph:
        def __init__(self, edges, num_nodes):
            self.edges, self.num_nodes = edges, num_nodes
            self.ndata, self.edata = {}, {}

    monkeypatch.setitem(sys.modules, "dgl", SimpleNamespace(graph=Graph))
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.ones(2, 3),
        y=torch.tensor([0, 1]),
        num_nodes=2,
    )
    graph = compat.to_dgl_graph(data, node_label="target")
    assert graph.num_nodes == 2 and set(graph.ndata) == {"target"}
    assert set(graph.edata) == {"edge_attr"}


# --------------------------------------------------------------------------- hpo


def test_hpo_template_falls_back_to_common_training_settings():
    args = SimpleNamespace(_method_config_keys={"seed", "_hidden", "values"}, values=[1], lr=0.1)
    args.seed, args.epochs = 3, 4
    assert hpo.template_space(args) == {"lr": [0.1], "epochs": [4]}
    assert hpo.template_space(SimpleNamespace(_method_config_keys={"lr"}, lr=0.2)) == {"lr": [0.2]}


def test_hpo_space_validation_and_grid(tmp_path):
    path = tmp_path / "space.json"
    for payload, message in [([1], "JSON object"), ({"_comment": "x"}, "empty")]:
        path.write_text(json.dumps(payload))
        with pytest.raises(click.BadParameter, match=message):
            hpo.load_space(path)
    assert hpo.grid({"b": [1, 2], "a": ["x"]}) == [{"a": "x", "b": 1}, {"a": "x", "b": 2}]
    table = hpo.box_table([["lr", 3]], ["name", "n"])
    assert table.splitlines()[1] == "| name | n |" and table.splitlines()[3] == "| lr   | 3 |"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, "unknown until the first trial finishes"),
        (float("nan"), "unknown until the first trial finishes"),
        (30, "30 s"),
        (600, "10.0 min"),
        (7200, "2.0 h"),
    ],
)
def test_hpo_duration_formatting(seconds, text):
    assert hpo.format_seconds(seconds) == text


def test_hpo_resolve_prefers_saved_space_and_trial_log_reloads(tmp_path):
    args = SimpleNamespace(
        save_path=str(tmp_path),
        method="kcenter",
        dataset="cora",
        reduction_rate=0.5,
        hpo_space=None,
        _method_config_keys=set(),
        lr=0.01,
    )
    folder = hpo.search_dir(args)
    assert folder == tmp_path.resolve() / "hpo/kcenter/cora_r0.5"
    folder.mkdir(parents=True)
    (folder / "space.json").write_text(json.dumps({"lr": [0.1, 0.2]}))
    path, space, created = hpo.resolve_space(args)
    assert (path, space, created) == (folder / "space.json", {"lr": [0.1, 0.2]}, False)
    shipped_args = SimpleNamespace(**{**vars(args), "method": "gdem"})
    path, space, created = hpo.resolve_space(shipped_args)
    assert path == hpo.SPACE_DIR / "gdem.json" and not created and space

    log = hpo.TrialLog(folder)
    assert log.mean_seconds() is None
    log.append({"trial": 0, "status": "complete", "seconds": 2.0})
    log.append({"trial": 1, "status": "failed", "seconds": 9.0})
    (folder / "trials.jsonl").write_text((folder / "trials.jsonl").read_text() + "\n")
    reloaded = hpo.TrialLog(folder)
    assert set(reloaded.records) == {0, 1} and reloaded.mean_seconds() == 2.0


def test_hpo_apply_and_reset_without_graph_reset():
    args = SimpleNamespace(epochs=4, eval_interval=2, hpo=True, wandb=True, lr=0.1)
    run_args = hpo.apply(args, {"epochs": 6})
    assert run_args.checkpoints == [-1, 1, 3, 5] and run_args.validation_only
    assert not run_args.hpo and not run_args.wandb and args.hpo and args.epochs == 4
    assert not hasattr(hpo.apply(args, {"lr": 1}), "checkpoints")
    graph = SimpleNamespace(feat_syn=1, adj_syn=2, labels_syn=3)
    hpo.reset_graph(graph)
    assert (graph.feat_syn, graph.adj_syn, graph.labels_syn) == (None, None, None)


def _search_args(tmp_path):
    args = quick_args(tmp_path, "random", "trans")
    args.hpo, args.hpo_yes, args.hpo_eval_runs = True, False, 1
    space = tmp_path / "space.json"
    space.write_text(json.dumps({"hidden": [8, 16]}))
    args.hpo_space = str(space)
    return args


def test_hpo_rejects_whole_graph_and_honours_declined_prompt(tmp_path, monkeypatch):
    args = _search_args(tmp_path)
    args.eval_whole = True
    with pytest.raises(click.BadParameter, match="eval_whole"):
        hpo.run_search(args, tiny_graph())
    args.eval_whole = False
    monkeypatch.setattr(click, "confirm", lambda *a, **k: False)
    assert hpo.run_search(args, tiny_graph()) is None
    assert not (hpo.search_dir(args) / "space_used.json").exists()


def test_hpo_reports_when_every_trial_fails(tmp_path, monkeypatch, capsys):
    args = _search_args(tmp_path)
    args.hpo_yes = True

    def broken(graph, run_args, runs):
        raise RuntimeError("no validation")

    monkeypatch.setattr(hpo, "validation_score", broken)
    assert hpo.run_search(args, tiny_graph()) is None
    output = capsys.readouterr().out
    assert "No trial completed." in output and "RuntimeError: no validation" in output
    assert not (hpo.search_dir(args) / "best.json").exists()


# --------------------------------------------------------------------------- script execution


@pytest.mark.parametrize(
    ("module", "argv", "exit_code"),
    [
        ("train_all", ["--help"], None),
        ("run_eval", ["--help"], None),
        ("visualization", ["--help"], 0),
        ("config", ["--help"], 0),
        ("doctor", ["--json"], None),
    ],
)
def test_modules_run_as_scripts(module, argv, exit_code, monkeypatch, capsys):
    import runpy
    from pathlib import Path

    import graphslim

    script = Path(graphslim.__file__).parent / f"{module}.py"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(script), *argv])
    if exit_code is None:
        runpy.run_path(str(script), run_name="__main__")
    else:
        with pytest.raises(SystemExit) as raised:
            runpy.run_path(str(script), run_name="__main__")
        assert raised.value.code == exit_code
    output = capsys.readouterr().out
    if module == "doctor":
        assert json.loads(output)["checks"]["pyg_gcn"]["status"] == "passed"
    else:
        assert "usage" in output.lower()


def test_reduced_graph_loader_transposes_row_major_coordinates(monkeypatch):
    coordinates = np.array([[0, 1], [1, 2], [2, 0]])
    adjacency = SimpleNamespace(nonzero=lambda: coordinates)
    monkeypatch.setattr(visualization.torch, "load", lambda path, map_location: adjacency)
    data = visualization.load_reduced_graph("unused.pt")
    assert data.edge_index.tolist() == [[0, 1, 2], [1, 2, 0]]


def test_hpo_reset_uses_graph_reset_hook_and_summary_counts_sum_without_item():
    graph = FakeGraph()
    hpo.reset_graph(graph)
    assert graph.resets == 1
    summary = tracking.graph_summary(SimpleNamespace(adj_syn=SimpleNamespace(sum=lambda: 6)))
    assert summary == {"edges": 6}

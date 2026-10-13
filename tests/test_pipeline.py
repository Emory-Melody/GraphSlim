from types import SimpleNamespace
import pytest

import graphslim.dataset as dataset
import graphslim.evaluation as evaluation
import graphslim.train_all as pipeline
import graphslim.utils as utils


@pytest.fixture
def setup_pipeline(monkeypatch):
    calls = {"seeds": [], "reductions": [], "evaluations": [], "resets": 0, "finished": 0}
    graph = SimpleNamespace()

    def reset():
        calls["resets"] += 1

    graph.reset = reset

    def reducer(method, setting, data, args):
        def reduce(data, verbose):
            calls["reductions"].append(args.seed)
            return data

        return SimpleNamespace(reduce=reduce)

    class Evaluator:
        def __init__(self, args):
            self.args = args

        def evaluate(self, data, model_type, reduced):
            calls["evaluations"].append((self.args.seed, reduced, model_type))
            return 0.7, 0.1

    def finish():
        calls["finished"] += 1

    tracker = SimpleNamespace(
        log_graph=lambda *a, **k: None, log=lambda *a, **k: None, finish=finish
    )
    monkeypatch.setattr(dataset, "get_dataset", lambda *a: graph)
    monkeypatch.setattr(utils, "seed_everything", calls["seeds"].append)
    monkeypatch.setattr(pipeline, "create_reducer", reducer)
    monkeypatch.setattr(evaluation, "Evaluator", Evaluator)
    monkeypatch.setattr(pipeline, "build_tracker", lambda args: tracker)
    args = SimpleNamespace(
        device="cpu",
        dataset="cora",
        method="gcond",
        setting="trans",
        seed=7,
        attack=None,
        load_path="data",
        run_reduction=3,
        eval_whole=False,
        eval_model="GCN",
        verbose=False,
    )
    return args, calls, graph


def test_repetitions_use_consecutive_seeds_and_preserve_args(setup_pipeline):
    args, calls, graph = setup_pipeline
    assert pipeline.run(args) == [(0.7, 0.1)] * 3
    assert calls["reductions"] == [7, 8, 9]
    assert calls["resets"] == 2
    assert calls["finished"] == 1
    assert args.seed == 7


@pytest.mark.parametrize("whole", [True, False])
def test_evaluation_only_does_not_construct_reducer(setup_pipeline, whole):
    args, calls, graph = setup_pipeline
    args.eval_whole = whole
    args.run_reduction = 0
    pipeline.run(args)
    assert calls["reductions"] == []
    assert calls["evaluations"] == [(7, not whole, "GCN")]


def test_tracker_finishes_after_failure(setup_pipeline, monkeypatch):
    args, calls, graph = setup_pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("reduction failed")

    monkeypatch.setattr(pipeline, "create_reducer", fail)
    with pytest.raises(RuntimeError, match="reduction failed"):
        pipeline.run(args)
    assert calls["finished"] == 1

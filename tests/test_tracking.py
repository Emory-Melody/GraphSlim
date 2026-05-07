from types import SimpleNamespace

import torch

from graphslim.tracking import NullTracker, build_tracker, graph_summary


def test_build_tracker_defaults_to_noop():
    tracker = build_tracker(SimpleNamespace(wandb=False))

    assert isinstance(tracker, NullTracker)
    tracker.log({"loss": 1.0})
    tracker.finish()


def test_graph_summary_for_reduced_graph():
    graph = SimpleNamespace(
        feat_syn=torch.ones(3, 2),
        adj_syn=torch.tensor(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
    )

    summary = graph_summary(graph)

    assert summary["nodes"] == 3
    assert summary["edges"] == 4
    assert summary["density"] > 0

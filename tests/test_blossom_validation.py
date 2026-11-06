"""Enable the matching solver's dual checks and compare to an external oracle."""

import networkx as nx
import numpy as np
import pytest

from graphslim.coarsening import utils


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("maxcardinality", [False, True])
def test_primal_dual_and_debug_checks_preserve_networkx_matching_optimum(
    monkeypatch, seed, maxcardinality
):
    messages = []
    monkeypatch.setattr(utils, "DEBUG", messages.append)
    monkeypatch.setattr(utils, "CHECK_DELTA", True)
    monkeypatch.setattr(utils, "CHECK_OPTIMUM", True)
    graph = nx.gnp_random_graph(15, 0.3 + 0.1 * (seed % 4), seed=seed)
    rng = np.random.default_rng(seed)
    for a, b in graph.edges:
        graph[a][b]["weight"] = int(rng.integers(-3, 15))
    mates = utils.maxWeightMatching(
        [(a, b, d["weight"]) for a, b, d in graph.edges(data=True)], maxcardinality=maxcardinality
    )
    pairs = [(a, b) for a, b in enumerate(mates) if a < b]
    oracle = nx.max_weight_matching(graph, maxcardinality=maxcardinality)
    assert sum(graph[a][b]["weight"] for a, b in pairs) == sum(
        graph[a][b]["weight"] for a, b in oracle
    )
    assert any(message.startswith("STAGE") for message in messages)
    if maxcardinality:
        assert len(pairs) == len(oracle)

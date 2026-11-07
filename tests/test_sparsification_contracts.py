# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Check edge reduction node, split, and checkpoint contracts."""

import numpy as np
import pytest
import torch

from graphslim.dataset import load_reduced
from graphslim.reduction import create_reducer
from graphslim.utils import seed_everything
from test_lightweight_methods import quick_args, tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "method",
    [
        "random_edge",
        "spanning_forest",
        "g_spar",
        "local_degree",
        "rank_degree",
        "scan",
        "t_spanner",
    ],
)
def test_edge_sparsifiers_keep_nodes_and_use_only_original_edges(tmp_path, method):
    graph = tiny_graph()
    args = quick_args(tmp_path, method, "trans")
    args.verbose, args.ts = True, 3
    seed_everything(17)
    original = graph.adj_full.toarray()
    reduced = create_reducer(method, "trans", graph, args).reduce(graph, verbose=True)
    adjacency = reduced.adj_syn.to_dense()
    assert adjacency.shape == original.shape
    assert torch.isfinite(adjacency).all()
    torch.testing.assert_close(adjacency, adjacency.T)
    assert torch.count_nonzero(adjacency) > 0
    assert not np.any((adjacency.numpy() != 0) & (original == 0))
    assert graph.feat_full.shape[0] == len(original)
    path = tmp_path / "reduced_graph" / method / f"adj_{args.dataset}_0.5_{args.seed}.pt"
    torch.testing.assert_close(torch.load(path, weights_only=True).to_dense(), adjacency)


@pytest.mark.parametrize("method", ["random_edge", "t_spanner"])
def test_inductive_edge_sparsifiers_cut_the_training_subgraph(tmp_path, method):
    graph = tiny_graph()
    args = quick_args(tmp_path, method, "ind")
    args.ts = 3
    seed_everything(5)
    original = graph.adj_train.toarray()
    reduced = create_reducer(method, "ind", graph, args).reduce(graph, verbose=True)
    adjacency = reduced.adj_syn.to_dense()
    assert args.edge_reduction is True
    assert adjacency.shape == original.shape == (16, 16)
    assert not np.any((adjacency.numpy() != 0) & (original == 0))
    assert 0 < reduced.reduced_edges <= original.sum() / 2
    assert reduced.reduction_metadata["graph"] == "train"
    # Evaluation falls back to the training features, which now match the adjacency.
    loaded_adj, loaded_feat, _ = load_reduced(args, graph)
    assert loaded_adj.shape[0] == loaded_feat.shape[0] == 16


def test_edge_sparsifier_refuses_node_reduction(tmp_path):
    graph = tiny_graph()
    args = quick_args(tmp_path, "random_edge", "trans")
    args.edge_reduction = False
    with pytest.raises(ValueError):
        create_reducer("random_edge", "trans", graph, args).reduce(graph, save=False)

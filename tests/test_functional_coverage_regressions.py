"""Regressions exposed by the real Dataset x Method x Application matrix.

Ported from an older snapshot. Cases that the current package fails because of
real defects (small-component coarsening, held-out labels in ``get_train_lcc``,
NaN means for empty classes, property reports for adjacency-only checkpoints)
are reported separately rather than encoded here; the removed
``LargeDataLoader(data=...)`` and ``get_eigh(root=...)`` options are not tested.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from graphslim.condensation.utils import get_embed_mean, get_train_lcc
from graphslim.dataset.loader import LargeDataLoader
from graphslim.dataset.utils import load_reduced, save_reduced
from test_lightweight_methods import tiny_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_edge_sparsifier_checkpoint_reloads_full_graph_with_original_features(tmp_path):
    graph = tiny_graph()
    args = SimpleNamespace(
        save_path=str(tmp_path),
        method="random_edge",
        attack=None,
        dataset="tiny",
        reduction_rate=0.5,
        seed=3,
        setting="trans",
        device="cpu",
        logger=logging.getLogger("functional-regressions"),
    )
    adjacency = torch.tensor(graph.adj_full.toarray())
    save_reduced(adjacency, args=args)
    adj, feat, labels = load_reduced(args, graph)
    torch.testing.assert_close(adj, adjacency)
    assert feat is graph.feat_full
    assert labels is graph.labels_train


def test_component_training_map_points_at_the_same_nodes_inside_the_component():
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 1])
    component = [1, 3, 4, 6]
    selected, local = get_train_lcc(component, [6, 3, 4], labels, 7, 2)
    assert sorted(selected) == [3, 4, 6]
    assert [component[i] for i in local] == selected


def test_class_means_are_unit_norm_for_present_classes():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    means = get_embed_mean(x, labels)
    torch.testing.assert_close(means[0], torch.tensor([2.0, 3.0]) / 13**0.5)
    torch.testing.assert_close(means.norm(dim=1), torch.ones(2))


@pytest.mark.parametrize("name", ["flickr", "reddit"])
def test_batch_loader_partitions_exactly_the_original_training_memberships(monkeypatch, name):
    import torch_geometric.datasets

    graph = tiny_graph()
    raw = Data(
        x=graph.feat_full,
        y=graph.labels_full,
        edge_index=graph.edge_index,
        train_mask=graph.train_mask,
        val_mask=graph.val_mask,
        test_mask=graph.test_mask,
    )

    class Dataset:
        num_classes = 2

        def __getitem__(self, index):
            return raw

    constructor = "Flickr" if name == "flickr" else "Reddit2"
    monkeypatch.setattr(torch_geometric.datasets, constructor, lambda **kw: Dataset())
    batches = LargeDataLoader(name=name, batch_size=6)
    batches.split_batch()
    assert (batches.n, batches.dim, batches.n_classes) == (32, 8, 2)
    assert int(batches.k) == 3
    observed = []
    for i in range(int(batches.k)):
        features, labels, adjacency = batches.get_batch(i)
        members = batches.split_idx[torch.as_tensor(batches.batch_labels) == i]
        observed.extend(members.tolist())
        assert len(features) == len(labels) == adjacency.shape[0] == len(members)
        torch.testing.assert_close(labels, graph.labels_full[members])
        dense = adjacency.to_dense()
        expected = torch.tensor(graph.adj_full[members.numpy()][:, members.numpy()].toarray())
        torch.testing.assert_close(dense, expected + torch.eye(len(members)))
        assert torch.isfinite(features).all()
    assert sorted(observed) == graph.idx_train.tolist()

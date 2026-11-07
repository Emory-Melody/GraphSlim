# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Tree encoding, reconstruction, graph unions, and frequency-weighted sampling."""

import pytest
import torch
from torch_geometric.data import Data

from graphslim.dataset import utils


def tree(weighted=False, label=0):
    data = Data(
        x=torch.tensor([[1.0, 0], [0, 1], [1, 1]]),
        edge_index=torch.tensor([[1, 2], [0, 0]]),
        y=torch.tensor([label]),
    )
    if weighted:
        data.edge_attr = torch.tensor([[0.5], [2.0]])
    return data


@pytest.mark.parametrize("weighted", [False, True])
def test_disjoint_union_offsets_edges_preserves_features_and_marks_roots(weighted):
    first, second = tree(weighted), tree(weighted)
    second.x = second.x + 3
    result = utils.disjointed_union([first, second], class_=0)
    assert result.x.shape == (6, 2) and result.edge_index.shape == (2, 4)
    torch.testing.assert_close(result.x, torch.cat([first.x, second.x]))
    torch.testing.assert_close(
        result.edge_index, torch.cat([first.edge_index, second.edge_index + 3], dim=1)
    )
    if weighted:
        torch.testing.assert_close(result.edge_attr, torch.cat([first.edge_attr, second.edge_attr]))
    else:
        assert result.edge_attr is None
    marked = utils.roots_to_embed(result)
    assert marked is result
    assert result.roots_to_embed.tolist() == [1, 0, 0, 1, 0, 0]
    assert result.y == 0


def test_single_node_trees_are_filtered_and_empty_union_is_none():
    single = Data(x=torch.ones(1, 2), edge_index=torch.empty(2, 0, dtype=torch.long), y=0)
    assert utils.disjointed_union([]) is None
    assert utils.disjointed_union([single]) is None
    result = utils.disjointed_union([single, tree()], class_=0)
    assert len(result.x) == 3


def test_disjoint_union_supports_scalar_edge_weights():
    first, second = tree(True), tree(True)
    first.edge_attr = first.edge_attr.flatten()
    second.edge_attr = second.edge_attr.flatten()
    result = utils.disjointed_union([first, second], class_=0)
    assert result.edge_attr.shape == (4,)
    torch.testing.assert_close(result.edge_attr, torch.tensor([0.5, 2, 0.5, 2]))


@pytest.mark.parametrize("case", ["classes", "features", "edge_features"])
def test_union_rejects_incompatible_tree_attributes(case):
    first, second = tree(weighted=True), tree(weighted=True)
    if case == "classes":
        second.y = torch.tensor([1])
    elif case == "features":
        second.x = torch.ones(3, 4)
    else:
        second.edge_attr = torch.ones(2, 3)
    with pytest.raises(AssertionError):
        utils.disjointed_union([first, second])


def test_root_detection_marks_only_sink_targets_and_skips_isolated_nodes():
    data = Data(x=torch.ones(5, 1), edge_index=torch.tensor([[1, 2, 3], [0, 0, 2]]))
    assert utils.roots_to_embed(data) is data
    assert data.roots_to_embed.tolist() == [1, 0, 0, 0, 0]


def test_root_detection_without_any_root_marks_nothing():
    # Regression: an empty root set used to build a float index tensor and crash.
    data = Data(x=torch.ones(4, 1), edge_index=torch.tensor([[0, 1], [1, 0]]))
    utils.roots_to_embed(data)
    assert data.roots_to_embed.tolist() == [0, 0, 0, 0]
    data.edge_index = torch.empty(2, 0, dtype=torch.long)
    utils.roots_to_embed(data)
    assert data.roots_to_embed.tolist() == [0] * 4


def test_canonical_encoding_parses_branches_and_restores_original_attributes():
    encoded = utils.prettify_canonical_label("1 10 5 2 20 $ 6 3 30 $ $")
    assert encoded == "<1,10><5,2,20>$<6,3,30>$$"
    nodes, edges, original = utils.parse_canonical_label(encoded)
    assert nodes == {0: "1", 1: "2", 2: "3"}
    assert edges == {(1, 0): "5", (2, 0): "6"}
    assert original == {0: "10", 1: "20", 2: "30"}
    data = utils.get_data(
        nodes,
        edges,
        original,
        {4: 1, 7: 2, 9: 3},
        {0.5: 5, 2.0: 6},
        {(1.0, 0.0): 10, (0.0, 1.0): 20, (1.0, 1.0): 30},
        1,
    )
    torch.testing.assert_close(data.x, tree().x)
    assert data.node_labels.tolist() == [4, 7, 9]
    assert data.edge_index.tolist() == [[1, 2], [0, 0]]
    assert data.edge_attr.tolist() == [0.5, 2.0]


def test_canonical_mapping_and_class_counts_keep_shared_patterns_explicit():
    labels = [["a", "b", "a"], ["b", "c"]]
    assert utils.canonical_label_to_naturals(labels) == {"a": 0, "b": 1, "c": 2}
    counts = utils.tree_class_ctr([0, 1], labels)
    assert counts == {"a": {0: 2}, "b": {0: 1, 1: 1}, "c": {1: 1}}
    assert utils.get_invalid_trees(counts) == ["b"]


def test_legacy_canonical_parser_and_malformed_labels():
    nodes, edges = utils.parse_canonical_label_bak("1<2,5>$<3,6>$$")
    assert nodes == {0: "1", 1: "2", 2: "3"}
    assert edges == {(1, 0): "5", (2, 0): "6"}
    for parser in [utils.parse_canonical_label, utils.parse_canonical_label_bak]:
        with pytest.raises(AssertionError, match="Parsing failed"):
            parser("not an encoded tree")


def test_label_preprocessing_preserves_features_and_shared_mapping():
    first, second = tree(), tree()
    first.y = torch.tensor([4, 7, 4])
    second.y = torch.tensor([7, 9, 4])
    second.x = second.x + 1
    processed, node_map, feature_map = utils.preprocess_dataset([first], [first, second])
    data = processed[0]
    torch.testing.assert_close(data.original_x, first.x)
    assert data.x.tolist() == [node_map[int(v)] for v in first.y]
    assert data.node_attr.tolist() == [feature_map[tuple(row.tolist())] for row in first.x]
    new = utils.preprocess_dataset_test([second], node_map, {}, feature_map)[0]
    assert new.x.tolist() == [node_map[int(v)] for v in second.y]


def test_sampling_uses_frequencies_requested_length_and_seed():
    first, second = tree(label=0), tree(label=1)
    dataset = [{0: {"freq": 1000, "data": first}}, {1: {"freq": 1, "data": second}}]
    a = utils.myDataset(dataset, dataset_len=200, seed=5)
    b = utils.myDataset(dataset, dataset_len=200, seed=5)
    assert len(a) == len(b) == 200
    labels_a = [int(a[i].y) for i in range(len(a))]
    labels_b = [int(b[i].y) for i in range(len(b))]
    assert labels_a == labels_b and labels_a.count(0) > 190
    loaders = [
        utils.get_dataloader({0: dataset[:1], 1: dataset[1:]}, dataset_len=6, batch_size=2),
        utils.get_dataloader([first, second], batch_size=2),
    ]
    assert sum(batch.num_graphs for batch in loaders[0]) == 6
    assert sum(batch.num_graphs for batch in loaders[1]) == 2

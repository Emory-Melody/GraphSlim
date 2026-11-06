"""Non-prefix train IDs expose full-graph versus training-row confusion (ported legacy test)."""

import numpy as np
import pytest
import torch

from graphslim.reduction import create_reducer
from test_lightweight_methods import quick_args, tiny_graph


@pytest.mark.parametrize("method", ["kcenter", "herding"])
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_selector_embeddings_have_the_same_node_order_as_the_selected_graph(
    tmp_path, monkeypatch, method, setting
):
    import graphslim.sparsification.model_based_coreset_base as module

    graph = tiny_graph()
    graph.idx_train = torch.tensor([9, 4, 17, 2, 29, 6, 25, 12])
    ids = graph.idx_train.numpy()
    graph.feat_train = graph.feat_full[ids]
    graph.labels_train = graph.labels_full[ids]
    graph.adj_train = graph.adj_full[np.ix_(ids, ids)]
    expected_features = graph.feat_train if setting == "ind" else graph.feat_full
    expected_adjacency = graph.adj_train if setting == "ind" else graph.adj_full

    class IdentityModel:
        def __init__(self, *args):
            pass

        def to(self, device):
            return self

        def fit_with_val(self, *args, **kwargs):
            assert kwargs["setting"] == setting

        def predict(self, features, adjacency):
            assert features is expected_features
            assert adjacency is expected_adjacency
            return features

    monkeypatch.setattr(module, "get_model_class", lambda name: IdentityModel)
    args = quick_args(tmp_path, method, setting)
    reducer = create_reducer(method, setting, graph, args)
    selected = reducer.select(expected_features)
    result = reducer.reduce(graph, save=False)
    torch.testing.assert_close(result.feat_syn, expected_features[selected])
    expected_labels = graph.labels_train if setting == "ind" else graph.labels_full
    torch.testing.assert_close(result.labels_syn, expected_labels[selected])
    expected_edges = expected_adjacency[np.ix_(selected, selected)].toarray()
    torch.testing.assert_close(result.adj_syn.to_dense(), torch.tensor(expected_edges).float())

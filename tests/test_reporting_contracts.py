"""Public evaluation reporting must retain predictions and support graph formats."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from graphslim.evaluation import utils


def test_f1_reporting_preserves_probability_matrix():
    labels = np.array([0, 1, 0])
    predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    before = predictions.copy()
    assert utils.calc_f1(labels, predictions, is_sigmoid=False) == (1, 1)
    np.testing.assert_array_equal(predictions, before)


def test_f1_reporting_thresholds_multilabel_probabilities():
    labels = np.array([[1, 0], [0, 1], [1, 1]])
    predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.9]])
    assert utils.calc_f1(labels, predictions.copy(), is_sigmoid=True) == (1, 1)
    wrong = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.1]])
    assert utils.calc_f1(labels, wrong, is_sigmoid=True) == (0, 0)


@pytest.mark.parametrize("dataset", ["flickr", "yelp"])
def test_graphsaint_reporting_uses_predicted_classes(dataset, capsys):
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    labels = torch.tensor([[1, 0], [0, 1]]) if dataset == "yelp" else torch.tensor([0, 1])
    before = output.clone()
    assert utils.evaluate(output, labels, SimpleNamespace(dataset=dataset)) is None
    assert "F1-micro= 1.0000 F1-macro= 1.0000" in capsys.readouterr().out
    if dataset == "flickr":
        torch.testing.assert_close(output, before)


def test_storage_reporting_counts_weighted_sparse_graphs():
    index = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([0.2, 0.2])
    adjacency = SparseTensor.from_edge_index(index, values, sparse_sizes=(3, 3))
    features = torch.ones((3, 2), dtype=torch.float32)
    labels = torch.tensor([0, 1, 0])
    expected_bytes = (
        index.numel() * index.element_size()
        + values.numel() * values.element_size()
        + features.numel() * features.element_size()
        + labels.numel() * labels.element_size()
    )
    assert utils.getsize_mb([adjacency, features, labels]) == pytest.approx(
        expected_bytes / 1024**2
    )

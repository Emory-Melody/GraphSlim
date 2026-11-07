# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Fraud AP preserves score ranking and drives both validation and test evaluation."""

import numpy as np
import pytest
import torch

from graphslim.config import get_args
from graphslim.utils import average_precision


def test_ap_ranks_positive_class_without_thresholding_or_mutation():
    # Positive examples rank second and third: AP = (1/2 + 2/3) / 2.
    margin = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    logits = torch.stack([torch.zeros_like(margin), margin], 1)
    labels = torch.tensor([0, 1, 1, 0])
    original = logits.detach().clone()
    assert average_precision(logits, labels) == pytest.approx(7 / 12)
    assert average_precision(logits.log_softmax(1), labels) == pytest.approx(7 / 12)
    # Arbitrary row offsets must not affect class odds or AP.
    shifted = logits + torch.tensor([100.0, -100.0, 20.0, 0.0])[:, None]
    assert average_precision(shifted, labels) == pytest.approx(7 / 12)
    torch.testing.assert_close(logits.detach(), original)


@pytest.mark.parametrize("shape", [(4,), (4, 1)])
def test_ap_accepts_continuous_scores_and_constant_prevalence(shape):
    labels = np.array([0, 1, 1, 0])
    assert average_precision(np.ones(shape), labels) == pytest.approx(0.5)
    assert average_precision(torch.ones(shape), torch.zeros(4)) == 0
    assert average_precision(torch.ones(shape), torch.ones(4)) == 1


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([[0, 1, 2]], [1]),
        ([float("nan")], [1]),
        ([0.2], [2]),
        ([], []),
        ([0.2], [0, 1]),
    ],
)
def test_ap_rejects_invalid_inputs(scores, labels):
    with pytest.raises(ValueError):
        average_precision(np.asarray(scores), np.asarray(labels))


@pytest.mark.parametrize("dataset", ["yelp", "amazon"])
def test_fraud_configuration_uses_ap_for_validation_and_test(tmp_path, dataset):
    args = get_args(["-D", dataset, "-M", "random", "-G", "-1", "--save_path", str(tmp_path)])
    assert args.metric is average_precision


@pytest.mark.parametrize(
    "model", ["GCN", "SGC", "APPNP", "Cheby", "GraphSage", "GAT", "GIN", "SGFormer"]
)
@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_all_backbones_use_ap_during_validation_and_test(tmp_path, model, setting):
    from graphslim.evaluation import Evaluator
    from test_lightweight_methods import quick_args, tiny_graph

    args = quick_args(tmp_path, "random", setting)
    args.eval_epochs = 3
    graph = tiny_graph()
    seen = []

    def ap_metric(output, labels):
        seen.append(output.detach().clone())
        return average_precision(output, labels)

    args.metric = ap_metric
    evaluator = Evaluator(args)
    validation, score = evaluator.test(graph, model_type=model, reduced=False, verbose=False)
    assert len(seen) >= 2  # training validation and final test both call the metric
    assert 0 <= validation <= 1
    assert score == pytest.approx(average_precision(seen[-1], graph.labels_test))

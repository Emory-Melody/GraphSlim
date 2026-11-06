"""GDEM feature initialization: a trained MLP drives synthetic features toward their labels.

Ported from the legacy suite. The legacy ``gdem_init_profile`` options (separate MLP epochs
and feature weight decay) no longer exist, so the test now checks the current behavior:
``init_feat`` optimizes the synthetic features against the classifier returned by
``mlp_trainer`` and ``mlp_trainer`` restores the best validation weights.
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from graphslim.condensation.gdem import GDEM
from test_legacy_reducer_integration import graph_fixture


def test_initialization_moves_synthetic_features_toward_target_classes(monkeypatch):
    agent = object.__new__(GDEM)
    agent.args = SimpleNamespace(lr_feat=0.01)
    agent.data = SimpleNamespace(labels_syn=torch.tensor([0, 1]))
    agent.device = "cpu"
    torch.manual_seed(12)
    agent.x_syn = nn.Parameter(torch.randn(2, 3))

    class FirstTwoFeatures(nn.Module):
        def forward(self, features):
            return F.log_softmax(features[:, :2], dim=1)

    calls = []

    def classifier(args, data, verbose):
        calls.append((args, data, verbose))
        return FirstTwoFeatures()

    monkeypatch.setattr(agent, "mlp_trainer", classifier)
    before = F.nll_loss(FirstTwoFeatures()(agent.x_syn.detach()), torch.tensor([0, 1]))
    output = agent.init_feat()
    after = F.nll_loss(FirstTwoFeatures()(output.detach()), torch.tensor([0, 1]))
    assert output is agent.x_syn and calls == [(agent.args, agent.data, False)]
    assert after < before * 0.1
    assert output[0, 0] > output[0, 1] and output[1, 1] > output[1, 0]


def test_mlp_trainer_restores_best_validation_weights():
    graph = graph_fixture()
    # Make classes linearly separable so validation accuracy reaches 1.
    graph.feat_full[:, 0] = graph.labels_full.float() * 4
    for split in ("train", "val", "test"):
        index = getattr(graph, f"idx_{split}")
        setattr(graph, f"feat_{split}", graph.feat_full[index])
    agent = object.__new__(GDEM)
    agent.device = "cpu"
    args = SimpleNamespace(hidden=8, dropout=0.0, lr=0.05, weight_decay=0.0)
    torch.manual_seed(0)
    model = agent.mlp_trainer(args, graph, verbose=False)
    model.eval()
    prediction = model(graph.feat_val).argmax(1)
    assert torch.equal(prediction, graph.labels_val)


def test_initialization_starts_from_xavier_values(monkeypatch):
    """Restored legacy check: with a gradient-free classifier the output is the Xavier start."""
    agent = object.__new__(GDEM)
    agent.args = SimpleNamespace(lr_feat=0.005)
    agent.data = SimpleNamespace(labels_syn=torch.tensor([0, 1]))
    agent.device = "cpu"
    # Uninitialized-looking contents must be overwritten, not optimized from.
    agent.x_syn = nn.Parameter(torch.full((2, 3), float("nan")))

    class ConstantClassifier(nn.Module):
        def forward(self, features):
            return F.log_softmax(features[:, :2] * 0, dim=1)

    monkeypatch.setattr(agent, "mlp_trainer", lambda args, data, verbose: ConstantClassifier())
    torch.manual_seed(12)
    initial = torch.empty(2, 3)
    nn.init.xavier_uniform_(initial)
    torch.manual_seed(12)
    torch.testing.assert_close(agent.init_feat().detach(), initial)

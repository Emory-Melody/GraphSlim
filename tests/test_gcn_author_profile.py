# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Released GDEM GCN initialization semantics.

Ported from the legacy suite. The opt-in input-dropout case is dropped because
``gcn_input_dropout`` no longer exists, and the transposed Glorot draw order was
replaced by the plain ``glorot`` initializer tested here.
"""

import torch

from graphslim.models import GCN
from graphslim.models.layers import GraphConvolution
from test_lightweight_methods import quick_args


def test_glorot_initializer_matches_xavier_draw_and_zero_bias():
    layer = GraphConvolution(4, 3, weight_initializer="glorot")
    torch.manual_seed(4)
    expected = torch.empty(4, 3)
    torch.nn.init.xavier_uniform_(expected)
    torch.manual_seed(4)
    layer.reset_parameters()
    assert torch.equal(layer.weight, expected)
    assert layer.bias.count_nonzero() == 0


def test_default_initializer_is_uniform_within_fan_out_bound():
    layer = GraphConvolution(5, 4)
    bound = 1 / 4**0.5
    assert layer.weight.abs().max() <= bound and layer.bias.abs().max() <= bound
    assert layer.bias.count_nonzero() > 0


def test_gcn_passes_configured_initializer_to_every_layer(tmp_path):
    args = quick_args(tmp_path, "gdem", "trans")
    args.nlayers, args.gcn_init = 3, "glorot"
    model = GCN(4, 6, 2, args)
    assert [layer.weight_initializer for layer in model.layers] == ["glorot"] * 3
    assert [tuple(layer.weight.shape) for layer in model.layers] == [(4, 6), (6, 6), (6, 2)]
    assert all(layer.bias.count_nonzero() == 0 for layer in model.layers)

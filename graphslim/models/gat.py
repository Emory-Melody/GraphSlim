# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
# License: MIT. Adapted from PyTorch Geometric; see ../THIRD_PARTY_NOTICES.md.
"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""

import torch.nn.functional as F
from torch import nn
import torch

from graphslim.models.gcn import BaseGNN
from torch_geometric.nn import GATConv


# from graphslim.models.layers import GATConv


class GAT(BaseGNN):
    """
    simple GAT model, one head and no edge weight
    only for evaluation
    """

    def __init__(self, nfeat, nhid, nclass, args, mode="train"):
        """Initialize GAT and its configuration.

        :param nfeat: Number of input features per node.
        :param nhid: Hidden representation width.
        :param nclass: Number of output classes.
        :param args: Configuration namespace containing the options used by this operation.
        :param mode: Model configuration mode; selects training or evaluation settings. Defaults to ``'train'``.
        """
        super(GAT, self).__init__(nfeat, nhid, nclass, args, mode)

        # Two attention layers: eight concatenated heads, then one output head.
        self.heads = 8
        self.output_heads = 1
        if mode in ["eval", "cross"]:
            self.nlayers = 2

        self.conv1 = GATConv(
            nfeat, nhid, heads=self.heads, dropout=self.dropout, bias=self.with_bias
        )

        self.conv2 = GATConv(
            nhid * self.heads,
            nclass,
            heads=self.output_heads,
            concat=False,
            dropout=self.dropout,
            bias=self.with_bias,
        )

        self.output = None
        self.best_model = None
        self.best_output = None
        self.initialize()

    def forward(self, x, adj, output_layer_features=False):
        """Apply graph attention layers to the provided feature and adjacency representation.

        :param x: Node feature matrix with one row per node.
        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param output_layer_features: Whether to also return intermediate layer features with the predictions. Defaults to ``False``.
        :returns: Class log probabilities; ``output_layer_features`` is accepted for interface compatibility.
        """
        if isinstance(adj, list):
            x_list = []
            for i in range(len(adj)):
                x_temp = F.dropout(x, p=self.dropout, training=self.training)
                x_temp = F.elu(self.conv1(x_temp, adj[i]))
                x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
                x_temp = self.conv2(x_temp, adj[i])
                x_list.append(x_temp)

            x = torch.cat(x_list, dim=0)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


# class GAT(BaseGNN):
#     def __init__(self, in_features, hidden_dim, num_classes, args, mode='train'):
#         super(GAT, self).__init__(in_features, hidden_dim, num_classes, args, mode)
#         num_heads = 8
#         dropout = args.dropout
#
#         self.conv1 = GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout)
#         self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout)
#
#     def forward(self, x, adj, output_layer_features=None):
#         x = F.elu(self.conv1(x, adj))
#         x = self.conv2(x, adj)
#         return F.softmax(x, dim=1)

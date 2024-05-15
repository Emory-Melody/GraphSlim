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
    '''
    simple GAT model, one head and no edge weight
    only for evaluation
    '''

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):
        super(GAT, self).__init__(nfeat, nhid, nclass, args, mode)

        if mode in ['eval', 'cross']:
            self.nlayers = 2
            self.heads = 8
            self.output_heads = 1

        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=self.heads,
            dropout=self.dropout,
            bias=self.with_bias)

        self.conv2 = GATConv(
            nhid * self.heads,
            nclass,
            heads=self.output_heads,
            concat=False,
            dropout=self.dropout,
            bias=self.with_bias)

        self.output = None
        self.best_model = None
        self.best_output = None
        self.initialize()

    def forward(self, x, adj, output_layer_features=False):
        if isinstance(adj, list):
            x_list = []
            for i in range(self.args.batch_adj):
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

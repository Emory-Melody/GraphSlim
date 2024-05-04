"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""

import torch.nn.functional as F
from torch import nn

from graphslim.models.gcn import BaseGNN
# from graphslim.evaluation import evaluate
# from torch_geometric.nn import GATConv

from graphslim.models.layers import GATConv


# from graphslim.dataset.convertor import Dpr2Pyg

class GAT(BaseGNN):

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):
        super(GAT, self).__init__(nfeat, nhid, nclass, args, mode)

        self.conv1 = GATConv(
            nfeat,
            nhid,
            dropout=self.dropout)

        self.conv2 = GATConv(
            nhid,
            nclass,
            dropout=self.dropout)
        self.layers = nn.ModuleList([self.conv1, self.conv2])

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj, output_layer_features=False):
        outputs = []
        for ix, layer in enumerate(self.layers):
            x = F.dropout(x, self.dropout, training=self.training)
            if ix != len(self.layers) - 1:
                x = F.elu(x)
            x = self.layers[ix](x, adj)
            if output_layer_features:
                outputs.append(x)

        if output_layer_features:
            return outputs
        x.view(-1, x.shape[-1])
        return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = F.dropout(x, self.dropout, training=self.training)
            if ix != len(self.layers) - 1:
                x = F.elu(x)
            x = self.layers[ix](x, adj)

        x.view(-1, x.shape[-1])
        x = x[:adjs[-1][2][1]]
        return F.log_softmax(x, dim=1)

    def forward_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = F.dropout(x, self.dropout, training=self.training)
            if ix != len(self.layers) - 1:
                x = F.elu(x)
            x = self.layers[ix](x, adj)

        x.view(-1, x.shape[-1])
        return F.log_softmax(x, dim=1)

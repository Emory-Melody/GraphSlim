"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim

from graphslim import utils
from graphslim.models.gcn import BaseGNN
# from graphslim.evaluation import evaluate
# from torch_geometric.nn import GATConv
from graphslim.models.layers import GATConv


# from graphslim.dataset.convertor import Dpr2Pyg

class GAT(BaseGNN):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
                 weight_decay=5e-4, with_bias=True, device=None, **kwargs):

        super(GAT, self).__init__(nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                                  with_relu=True, with_bias=True, with_bn=False, device=device)

        if 'dataset' in kwargs:
            if kwargs['dataset'] in ['ogbn-arxiv']:
                dropout = 0.7  # arxiv
            elif kwargs['dataset'] in ['reddit']:
                dropout = 0.05;
                self.dropout = 0.1;
                self.weight_decay = 5e-4
                # self.weight_decay = 5e-2; dropout=0.05; self.dropout=0.1
            elif kwargs['dataset'] in ['citeseer']:
                dropout = 0.7
                self.weight_decay = 5e-4
            elif kwargs['dataset'] in ['flickr']:
                dropout = 0.8
                # nhid=8; heads=8
                # self.dropout=0.1
            else:
                dropout = 0.7  # cora, citeseer, reddit
        else:
            dropout = 0.7
        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x, adj):
        # x, edge_index = data.x, data.edge_index
        x, edge_index = x, adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        # print(self.conv1.att_l.sum())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


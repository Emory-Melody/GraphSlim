"""multiple transformaiton and multiple propagation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse

from graphslim.models.base import BaseGNN
from graphslim.models.layers import MyLinear


class SGC(BaseGNN):
    '''
    multiple transformation layers
    '''

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):

        """nlayers indicates the number of propagations"""
        super(SGC, self).__init__(nfeat, nhid, nclass, args, mode)

        if self.ntrans == 1:
            self.layers.append(MyLinear(nfeat, nclass))
        else:
            self.layers.append(MyLinear(nfeat, nhid))
            if self.with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(self.ntrans - 2):
                if self.with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
                self.layers.append(MyLinear(nhid, nhid))
            self.layers.append(MyLinear(nhid, nclass))

    def forward(self, x, adj, output_layer_features=False):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for i in range(self.nlayers):
            if type(adj) == torch.Tensor:
                x = adj @ x
            else:
                x = torch_sparse.matmul(adj, x)

        x = x.reshape(-1, x.shape[-1])
        return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for ix, (adj, _, size) in enumerate(adjs):
            if type(adj) == torch.Tensor:
                x = adj @ x
            else:
                x = torch_sparse.matmul(adj, x)

        return F.log_softmax(x, dim=1)

    def forward_syn(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for ix, (adj) in enumerate(adjs):
            if type(adj) == torch.Tensor:
                x = adj @ x
            else:
                x = torch_sparse.matmul(adj, x)

        return F.log_softmax(x, dim=1)

# class SGC(BaseGNN):
#
#     def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
#                  with_relu=True, with_bias=True, with_bn=False, device=None):
#         super(SGC, self).__init__(nfeat, nhid, nclass, nlayers, dropout, lr, weight_decay,
#                                   with_relu, with_bias, with_bn, device=device)
#
#         self.conv = GraphConvolution(nfeat, nclass, with_bias=with_bias)
#         self.nlayers = nlayers
#
#         if not with_relu:
#             self.weight_decay = 0
#         else:
#             self.weight_decay = weight_decay
#         self.with_relu = with_relu
#         if with_bn:
#             print('Warning: SGC does not have bn!!!')
#
#     def forward(self, x, adj, output_layer_features=False):
#         weight = self.conv.weight
#         bias = self.conv.bias
#         x = torch.mm(x, weight)
#         for i in range(self.nlayers):
#             if isinstance(adj, list) or len(adj.shape) == 3:
#                 # only synthetic graph use batched adj
#                 adj = torch.as_tensor(adj)
#                 x = torch.matmul(adj, x)
#             else:
#                 x = torch.spmm(adj, x)
#         x = x + bias
#         return F.log_softmax(x, dim=1)
#
#     def forward_sampler(self, x, adjs):
#         weight = self.conv.weight
#         bias = self.conv.bias
#         x = torch.mm(x, weight)
#         for ix, (adj, _, size) in enumerate(adjs):
#             x = torch_sparse.matmul(adj, x)
#         x = x + bias
#
#         return F.log_softmax(x, dim=1)
#
#     def forward_syn(self, x, adjs):
#         weight = self.conv.weight
#         bias = self.conv.bias
#         x = torch.mm(x, weight)
#         for ix, (adj) in enumerate(adjs):
#             if type(adj) == torch.Tensor:
#                 x = adj @ x
#             else:
#                 x = torch_sparse.matmul(adj, x)
#         x = x + bias
#
#         return F.log_softmax(x, dim=1)
#
#     def initialize(self):
#         """Initialize parameters of GCN.
#         """
#         self.conv.reset_parameters()
#         if self.with_bn:
#             for bn in self.bns:
#                 bn.reset_parameters()
#

"""multiple transformaiton and multiple propagation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_sparse import SparseTensor

from graphslim.models.base import BaseGNN
from graphslim.models.layers import MyLinear


class APPNP(BaseGNN):

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):

        super(APPNP, self).__init__(nfeat, nhid, nclass, args, mode)


        self.layers = nn.ModuleList([])
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

        self.sparse_dropout = SparseDropout(dprob=0)

        activation_functions = {
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'relu': F.relu,
            'linear': lambda x: x,
            'softplus': F.softplus,
            'leakyrelu': F.leaky_relu,
            'relu6': F.relu6,
            'elu': F.elu
        }
        self.activation = activation_functions.get(args.activation)

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = self.activation(x)
                x = F.dropout(x, self.dropout, training=self.training)

        h = x
        # here nlayers means K
        for i in range(self.nlayers):
            # adj_drop = self.sparse_dropout(adj, training=self.training)
            adj_drop = adj
            if isinstance(adj_drop, SparseTensor):
                x = torch_sparse.matmul(adj_drop, x)
            else:
                x = torch.spmm(adj_drop, x)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        h = x
        for ix, (adj, _, size) in enumerate(adjs):
            adj_drop = adj
            h = h[: size[1]]
            x = torch_sparse.matmul(adj_drop, x)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        self.kprob = 1 - dprob

    def forward(self, x, training):
        if training:
            mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
            rc = x._indices()[:, mask]
            val = x._values()[mask] * (1.0 / self.kprob)
            return torch.sparse.FloatTensor(rc, val, x.size())
        else:
            return x

# class APPNPRich(BaseGNN):
#     '''
#     two transformation layer
#     '''
#
#     def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, alpha=0.1,
#                  activation="relu", with_relu=True, with_bias=True, with_bn=False, device=None):
#
#         super(APPNPRich, self).__init__(nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
#                                         with_relu=True, with_bias=True, with_bn=False, device=device)
#
#         self.alpha = alpha
#         activation_functions = {
#             'sigmoid': F.sigmoid,
#             'tanh': F.tanh,
#             'relu': F.relu,
#             'linear': lambda x: x,
#             'softplus': F.softplus,
#             'leakyrelu': F.leaky_relu,
#             'relu6': F.relu6,
#             'elu': F.elu
#         }
#         self.activation = activation_functions.get(activation)
#
#         if with_bn:
#             self.bns = torch.nn.ModuleList()
#             self.bns.append(nn.BatchNorm1d(nhid))
#
#         self.layers = nn.ModuleList([])
#         self.layers.append(MyLinear(nfeat, nhid))
#         self.layers.append(MyLinear(nhid, nclass))
#
#         # if nlayers == 1:
#         #     self.layers.append(nn.Linear(nfeat, nclass))
#         # else:
#         #     self.layers.append(nn.Linear(nfeat, nhid))
#         #     for i in range(nlayers-2):
#         #         self.layers.append(nn.Linear(nhid, nhid))
#         #     self.layers.append(nn.Linear(nhid, nclass))
#
#         self.nlayers = nlayers
#         self.dropout = dropout
#         self.lr = lr
#
#         self.sparse_dropout = SparseDropout(dprob=0)
#
#     def forward(self, x, adj, output_layer_features=None):
#         for ix, layer in enumerate(self.layers):
#             x = layer(x)
#             if ix != len(self.layers) - 1:
#                 x = self.bns[ix](x) if self.with_bn else x
#                 # x = F.relu(x)
#                 x = self.activation(x)
#                 x = F.dropout(x, self.dropout, training=self.training)
#
#         h = x
#         # here nlayers means K
#         for i in range(self.nlayers):
#             # adj_drop = self.sparse_dropout(adj, training=self.training)
#             adj_drop = adj
#             if isinstance(adj_drop, SparseTensor):
#                 x = torch_sparse.matmul(adj_drop, x)
#             else:
#                 x = torch.spmm(adj_drop, x)
#             x = x * (1 - self.alpha)
#             x = x + self.alpha * h
#
#         if self.multi_label:
#             return torch.sigmoid(x)
#         else:
#             return F.log_softmax(x, dim=1)
#
#     def forward_sampler(self, x, adjs):
#         for ix, layer in enumerate(self.layers):
#             x = layer(x)
#             if ix != len(self.layers) - 1:
#                 x = self.bns[ix](x) if self.with_bn else x
#                 x = self.activation(x)
#                 x = F.dropout(x, self.dropout, training=self.training)
#
#         h = x
#         for ix, (adj, _, size) in enumerate(adjs):
#             adj_drop = adj
#             h = h[: size[1]]
#             x = torch_sparse.matmul(adj_drop, x)
#             x = x * (1 - self.alpha)
#             x = x + self.alpha * h
#
#         if self.multi_label:
#             return torch.sigmoid(x)
#         else:
#             return F.log_softmax(x, dim=1)

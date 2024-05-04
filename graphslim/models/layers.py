import math
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor, set_diag, matmul
from torch_geometric.utils import dense_to_sparse


class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.zeros(out_features))  # change this line
        else:
            self.bias = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """ Graph Convolutional Layer forward function
        """
        if isinstance(adj, SparseTensor):
            x = torch.mm(x, self.weight)
            x = matmul(adj, x)
        else:
            x = x.reshape(-1, x.shape[-1])
            x = torch.mm(x, self.weight)
            x = x.reshape(-1, adj.shape[1], x.shape[-1])
            x = adj @ x
        x.view(-1, x.shape[-1])
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels,
                 negative_slope=0.2, dropout=0.0,
                 bias=True):
        super(GATConv, self).__init__(node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()
        self.edge_weight = None

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def set_adj(self, rows, cols, batch=None):
        self.rows = rows
        self.cols = cols
        self.batch = batch
        if batch is not None:
            batch_size = batch.max() + 1
            self.adj_t = torch.zeros(size=(batch_size, self.n_syn, self.n_syn), device=self.device)

    def forward(self, x, adj_t):
        C = self.out_channels
        x_l = x_r = self.lin(x).view(-1, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        if isinstance(adj_t, SparseTensor):
            row, col, _ = adj_t.coo()
            edge_index = torch.vstack([row, col])
            out = self.propagate(edge_index, x=(x_l, x_r),
                                 alpha=(alpha_l, alpha_r))
        else:
            edge_index, _ = dense_to_sparse(adj_t)
            out = self.propagate(edge_index, x=(x_l, x_r),
                                 alpha=(alpha_l, alpha_r))
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, alpha_j, alpha_i,
                index, ptr, size_i) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class SageConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SageConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, adj_t):
        if isinstance(adj_t, SparseTensor):
            h = matmul(adj_t, x, reduce='add')
            out = self.lin_r(h)  ####################
            out += self.lin_r(x[:adj_t.size(0)])
        else:
            h1 = self.lin_r(x)
            h1 = h1.reshape(-1, adj_t.shape[1], h1.shape[1])
            h2 = adj_t @ x.reshape(-1, adj_t.shape[1], x.shape[1])
            h2 = h2.reshape(-1, h2.shape[2])
            h2 = self.lin_r(h2)  ##############################
            h2 = h2.reshape(-1, adj_t.shape[1], h2.shape[1])
            out = h1 + h2
            out = out.reshape(-1, out.shape[2])
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ChebConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True, single_param=True, K=2):
        """set single_param to True to alleivate the overfitting issue"""
        super(ChebConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lins = torch.nn.ModuleList([
            MyLinear(in_features, out_features, with_bias=False) for _ in range(K)])
        # self.lins = torch.nn.ModuleList([
        #    MyLinear(in_features, out_features, with_bias=True) for _ in range(K)])
        if with_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.single_param = single_param
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, input, adj, size=None):
        """ Graph Convolutional Layer forward function
        """
        # support = torch.mm(input, self.weight_l)
        x = input
        Tx_0 = x[:size[1]] if size is not None else x
        Tx_1 = x  # dummy
        output = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            if isinstance(adj, SparseTensor):
                Tx_1 = matmul(adj, x)
            else:
                Tx_1 = torch.spmm(adj, x)

            if self.single_param:
                output = output + self.lins[0](Tx_1)
            else:
                output = output + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            if self.single_param:
                lin = self.lins[0]
            if isinstance(adj, SparseTensor):
                Tx_2 = matmul(adj, Tx_1)
            else:
                Tx_2 = torch.spmm(adj, Tx_1)
            Tx_2 = 2. * Tx_2 - Tx_0
            output = output + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class MyLinear(torch.nn.Module):
    """Simple Linear layer, modified from https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

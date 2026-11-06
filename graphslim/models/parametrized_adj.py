# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGE(nn.Module):
    """Parameterized adjacency estimator from pairs of synthetic node features."""

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        """Initialize PGE and its configuration.

        :param nfeat: Number of input features per node.
        :param nnodes: Number of nodes used to construct candidate edge pairs.
        :param nhid: Hidden representation width. Defaults to ``128``.
        :param nlayers: Number of layers. Defaults to ``3``.
        :param device: Target PyTorch device, such as ``cpu`` or ``cuda:0``. Defaults to ``None``.
        :param args: Configuration namespace containing the options used by this operation. Defaults to ``None``.
        """
        super(PGE, self).__init__()
        if args.dataset in ["ogbn-arxiv", "arxiv", "flickr"]:
            nhid = 256
        if args.dataset in ["reddit"]:
            nhid = 256
            if args.reduction_rate == 0.01:
                nhid = 128
            nlayers = 3
            # nhid = 128

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat * 2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        # edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        X, Y = np.meshgrid(np.arange(nnodes), np.arange(nnodes), copy=False)
        edge_index = np.column_stack((X.ravel(), Y.ravel()))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x, inference=False):
        """Predict a symmetric adjacency from pairs of synthetic node features.

        :param x: Node feature matrix with one row per node.
        :param inference: Compatibility flag; the current forward computation is identical for both values. Defaults to ``False``.
        :returns: Dense adjacency probability matrix with its diagonal removed.
        """
        if self.args.dataset == "reddit" and self.args.reduction_rate >= 0.01:
            edge_index = self.edge_index
            n_part = 5
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
            edge_embed = []
            for idx in splits:
                tmp_edge_embed = torch.cat([x[edge_index[0][idx]], x[edge_index[1][idx]]], axis=1)
                for ix, layer in enumerate(self.layers):
                    tmp_edge_embed = layer(tmp_edge_embed)
                    if ix != len(self.layers) - 1:
                        tmp_edge_embed = self.bns[ix](tmp_edge_embed)
                        tmp_edge_embed = F.relu(tmp_edge_embed)
                edge_embed.append(tmp_edge_embed)
            edge_embed = torch.cat(edge_embed)
        else:
            edge_index = self.edge_index
            edge_embed = torch.cat([x[edge_index[0]], x[edge_index[1]]], axis=1)
            for ix, layer in enumerate(self.layers):
                edge_embed = layer(edge_embed)
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)

        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        """Evaluate the graph estimator with gradients disabled.

        :param x: Node feature matrix with one row per node.
        :returns: Predicted dense synthetic adjacency.
        """
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""

        def weight_reset(m):
            """Reset linear and batch normalization submodule parameters.

            :param m: PyTorch submodule to reset.
            """
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

        self.apply(weight_reset)

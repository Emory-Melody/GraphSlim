# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Graph Isomorphism Network for node prediction on original and reduced graphs."""

import torch
from torch import nn
from torch_sparse import SparseTensor

from graphslim.models.base import BaseGNN


class GINConvolution(nn.Module):
    r"""Apply an MLP to ``A @ X + (1 + eps) * X`` without degree normalization.

    Adjacency rows identify receiving nodes. Dense, batched dense, native Torch
    sparse, and ``torch_sparse.SparseTensor`` inputs retain their edge weights.
    Existing diagonal entries are part of ``A``, as in PyG's GIN operators.
    """

    def __init__(self, in_features, out_features, hidden_features=None, eps=0.0, train_eps=False):
        """Initialize the two-layer MLP and the self-feature coefficient.

        :param in_features: Number of input features per node.
        :param out_features: Number of output features per node.
        :param hidden_features: MLP hidden width; ``None`` uses ``out_features``.
        :param eps: Initial offset in the self-feature multiplier ``1 + eps``.
            Defaults to ``0.0``.
        :param train_eps: Whether to learn ``eps`` instead of storing it as a
            fixed buffer. Defaults to ``False``.
        """
        super().__init__()
        width = out_features if hidden_features is None else hidden_features
        self.nn = nn.Sequential(
            nn.Linear(in_features, width), nn.ReLU(), nn.Linear(width, out_features)
        )
        self.initial_eps = float(eps)
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps]))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps]))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset both MLP layers and the initial epsilon offset."""
        for layer in self.nn:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        with torch.no_grad():
            self.eps.fill_(self.initial_eps)

    def forward(self, x, adj):
        """Sum neighboring features, add self features, and apply the MLP.

        :param x: Node features with shape ``(nodes, in_features)`` or
            ``(batch, nodes, in_features)``. Batched dense adjacency also accepts
            flattened features with shape ``(batch * nodes, in_features)``.
        :param adj: Dense adjacency, a batch of dense adjacencies, a native
            Torch sparse matrix, or a ``torch_sparse.SparseTensor``. Rows
            identify receiving nodes; edge weights are retained.
        :returns: Transformed node features with width ``out_features``;
            flattened batched inputs produce flattened outputs.
        """
        flatten_output = False
        if isinstance(adj, SparseTensor):
            neighbors = adj.matmul(x)
        elif adj.layout != torch.strided:
            neighbors = torch.sparse.mm(adj, x)
        else:
            if adj.dim() == 3 and x.dim() == 2:
                x = x.reshape(-1, adj.shape[-1], x.shape[-1])
                flatten_output = True
            neighbors = adj @ x
        root = x[..., : neighbors.shape[-2], :]
        output = self.nn(neighbors + (1 + self.eps) * root)
        return output.reshape(-1, output.shape[-1]) if flatten_output else output


class GIN(BaseGNN):
    """GIN classifier using sum aggregation and two-layer MLPs.

    The shared evaluator trains on reduced node graphs, selects model weights on
    the original validation partition, and predicts on original graph nodes.
    Graph pooling and graph-level classification are outside this model's API.
    """

    normalize_adjacency = False

    def __init__(self, nfeat, nhid, nclass, args, mode="train"):
        """Build GIN layers with optional hidden-layer batch normalization.

        :param nfeat: Number of input features per node.
        :param nhid: Hidden feature width and hidden width of each layer's MLP.
        :param nclass: Number of output classes.
        :param args: Shared model configuration, including ``nlayers`` (at least
            one), ``with_bn``, dropout, optimizer and metric settings. Optional
            ``gin_eps`` and ``gin_train_eps`` default to ``0.0`` and ``False``.
        :param mode: Mode passed to ``BaseGNN``. ``eval`` disables dropout and
            sets evaluation weight decay; ``attack`` sets the NLL loss.
            Defaults to ``'train'``.
        :raises ValueError: If ``args.nlayers`` is less than one.
        """
        super().__init__(nfeat, nhid, nclass, args, mode)
        if self.nlayers < 1:
            raise ValueError("GIN requires at least one graph layer")
        self.bns = nn.ModuleList()
        dimensions = [nfeat] + [nhid] * (self.nlayers - 1) + [nclass]
        for index in range(self.nlayers):
            self.layers.append(
                GINConvolution(
                    dimensions[index],
                    dimensions[index + 1],
                    hidden_features=nhid,
                    eps=getattr(args, "gin_eps", 0.0),
                    train_eps=getattr(args, "gin_train_eps", False),
                )
            )
            if self.with_bn and index < self.nlayers - 1:
                self.bns.append(nn.BatchNorm1d(nhid))

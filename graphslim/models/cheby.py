import torch
import torch.nn as nn

from graphslim.models.base import BaseGNN
from graphslim.models.layers import ChebConvolution


class Cheby(BaseGNN):
    """Graph classifier using Chebyshev polynomial convolutions."""

    def __init__(self, nfeat, nhid, nclass, args, mode="train"):
        """Initialize Cheby and its configuration.

        :param nfeat: Number of input features per node.
        :param nhid: Hidden representation width.
        :param nclass: Number of output classes.
        :param args: Configuration namespace containing the options used by this operation.
        :param mode: Model configuration mode; selects training or evaluation settings. Defaults to ``'train'``.
        """
        super(Cheby, self).__init__(nfeat, nhid, nclass, args, mode)

        with_bias = self.with_bias

        if self.nlayers == 1:
            self.layers.append(ChebConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if self.with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(ChebConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(self.nlayers - 2):
                self.layers.append(ChebConvolution(nhid, nhid, with_bias=with_bias))
                if self.with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(ChebConvolution(nhid, nclass, with_bias=with_bias))

        # self.lin = MyLinear(nhid, nclass, with_bias=True)

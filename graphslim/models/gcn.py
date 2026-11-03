import torch.nn as nn

from graphslim.models.base import BaseGNN
from graphslim.models.layers import GraphConvolution
from graphslim.utils import *


class GCN(BaseGNN):
    """Graph convolutional classifier supporting dense and sparse adjacency."""

    def __init__(self, nfeat, nhid, nclass, args, mode="train"):
        """Initialize GCN and its configuration.

        :param nfeat: Number of input features per node.
        :param nhid: Hidden representation width.
        :param nclass: Number of output classes.
        :param args: Configuration namespace containing the options used by this operation.
        :param mode: Model configuration mode; selects training or evaluation settings. Defaults to ``'train'``.
        """
        super(GCN, self).__init__(nfeat, nhid, nclass, args, mode)
        initializer = getattr(args, "gcn_init", None)

        if self.nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, weight_initializer=initializer))
        else:
            if self.with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, weight_initializer=initializer))
            for i in range(self.nlayers - 2):
                self.layers.append(GraphConvolution(nhid, nhid, weight_initializer=initializer))
                if self.with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, weight_initializer=initializer))

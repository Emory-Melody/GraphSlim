import torch.nn as nn

from graphslim.models.base import BaseGNN
from graphslim.models.layers import GraphConvolution
from graphslim.utils import *


class GCN(BaseGNN):
    def __init__(self, nfeat, nhid, nclass, args, mode='train'):
        super(GCN, self).__init__(nfeat, nhid, nclass, args, mode)

        if self.nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass))
        else:
            if self.with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid))
            for i in range(self.nlayers - 2):
                self.layers.append(GraphConvolution(nhid, nhid))
                if self.with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass))

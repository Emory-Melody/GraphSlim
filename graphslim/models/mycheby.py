import torch
import torch.nn as nn

from graphslim.models.layers import ChebConvolution


class Cheby(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):

        super(Cheby, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(ChebConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(ChebConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers - 2):
                self.layers.append(ChebConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(ChebConvolution(nhid, nclass, with_bias=with_bias))

        # self.lin = MyLinear(nhid, nclass, with_bias=True)

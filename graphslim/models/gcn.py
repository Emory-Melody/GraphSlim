from copy import deepcopy

import torch.nn as nn
import torch.optim as optim

from graphslim.models.layers import GraphConvolution
from graphslim.models.base import BaseGNN
from graphslim.utils import *




class GCN(BaseGNN):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):
        super(GCN, self).__init__(nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                                  with_relu=True, with_bias=True, with_bn=False, device=device)
        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers - 2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

    # def forward_sampler_syn(self, x, adjs):
    #     for ix, (adj) in enumerate(adjs):
    #         x = self.layers[ix](x, adj)
    #         if ix != len(self.layers) - 1:
    #             x = self.bns[ix](x) if self.with_bn else x
    #             if self.with_relu:
    #                 x = F.relu(x)
    #             x = F.dropout(x, self.dropout, training=self.training)
    #
    #     if self.multi_label:
    #         return torch.sigmoid(x)
    #     else:
    #         return F.log_softmax(x, dim=1)

    # def _train_with_val2(self, labels, idx_train, idx_val, train_iters, verbose):
    #     if verbose:
    #         print('=== training gcn model ===')
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #
    #     best_loss_val = 100
    #     best_acc_val = 0
    #
    #     for i in range(train_iters):
    #         if i == train_iters // 2:
    #             lr = self.lr*0.1
    #             optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)
    #
    #         self.train()
    #         optimizer.zero_grad()
    #         output = self.forward(self.features, self.adj_norm)
    #         loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #         loss_train.backward()
    #         optimizer.step()
    #
    #         if verbose and i % 10 == 0:
    #             print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
    #
    #         self.eval()
    #         output = self.forward(self.features, self.adj_norm)
    #         loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #         acc_val = utils.accuracy(output[idx_val], labels[idx_val])
    #
    #         if acc_val > best_acc_val:
    #             best_acc_val = acc_val
    #             self.output = output
    #             weights = deepcopy(self.state_dict())
    #
    #     if verbose:
    #         print('=== picking the best model according to the performance on validation ===')
    #     self.load_state_dict(weights)

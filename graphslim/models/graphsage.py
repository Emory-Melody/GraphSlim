from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import NeighborSampler

from graphslim.models.base import BaseGNN
from graphslim.models.layers import SageConvolution
from graphslim.utils import is_sparse_tensor, normalize_adj_tensor, to_tensor, accuracy


class GraphSage(BaseGNN):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GraphSage, self).__init__(nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                                        with_relu=True, with_bias=True, with_bn=False, device=device)
        self.layers = nn.ModuleList([])

        if self.nlayers == 1:
            self.layers.append(SageConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers - 2):
                self.layers.append(SageConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nhid, nclass, with_bias=with_bias))

    def fit_with_val(self, data, train_iters=200, verbose=False,
                     normadj=True, normfeat=True, setting='trans', reduced=False, reindex=False,
                     **kwargs):

        self.initialize()
        # data for training
        if reduced:
            adj, features, labels, labels_val = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn, data.labels_val,
                                                          device=self.device)
        elif setting == 'trans':
            adj, features, labels, labels_val = to_tensor(data.adj_full, data.feat_full, data.labels_train,
                                                          data.labels_val, device=self.device)
        else:
            adj, features, labels, labels_val = to_tensor(data.adj_train, data.feat_train, data.labels_train,
                                                          data.labels_val, device=self.device)
        if normadj:
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        if normfeat:
            features = F.normalize(features, p=1, dim=1)

        if len(data.labels_full.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        if reduced:
            reindex = True

        if verbose:
            print('=== training GNN model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        # data for validation
        if setting == 'ind':
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = to_tensor(feat_full, adj_full, device=self.device)
        if normadj:
            adj_full = normalize_adj_tensor(adj_full, sparse=is_sparse_tensor(adj_full))
        if normfeat:
            feat_full = F.normalize(feat_full, p=1, dim=1)
        if adj.density() > 0.5:  # if the weighted graph is too dense, we need a larger neighborhood size
            sizes = [30, 20]
        else:
            sizes = [5, 5]
        if reduced:
            node_idx = torch.arange(data.labels_syn.size(0), device=self.device)
        elif setting == 'ind':
            node_idx = torch.arange(data.labels_train.size(0), device=self.device)
        else:
            node_idx = torch.arange(data.labels_full.size(0), device=self.device)

        train_loader = NeighborSampler(adj,
                                       node_idx=node_idx,
                                       sizes=sizes, batch_size=len(node_idx),
                                       num_workers=0, return_e_id=False,
                                       num_nodes=adj.size(0),
                                       shuffle=True)

        best_acc_val = 0
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr * 0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            # optimizer.zero_grad()
            # output = self.forward(self.features, self.adj_norm)
            # loss_train = self.loss(output, labels)
            # loss_train.backward()
            # optimizer.step()

            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(self.device) for adj in adjs]
                optimizer.zero_grad()
                out = self.forward_sampler(features[n_id], adjs)
                loss_train = F.nll_loss(out, labels[n_id[:batch_size]])
                loss_train.backward()
                optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full)
                if setting == 'ind':
                    # loss_val = F.nll_loss(output, labels_val)
                    acc_val = accuracy(output, labels_val)
                else:
                    # loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        return best_acc_val

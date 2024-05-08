from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborSampler

from graphslim.dataset.convertor import dense2sparsetensor
from graphslim.models.base import BaseGNN
from graphslim.models.layers import SageConvolution
from graphslim.utils import is_sparse_tensor, normalize_adj_tensor, to_tensor, accuracy


class GraphSage(BaseGNN):

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):

        super(GraphSage, self).__init__(nfeat, nhid, nclass, args, mode)
        with_bn = self.with_bn

        if self.nlayers == 1:
            self.layers.append(SageConvolution(nfeat, nclass))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nfeat, nhid))
            for i in range(self.nlayers - 2):
                self.layers.append(SageConvolution(nhid, nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nhid, nclass))

    def fit_with_val(self, data, train_iters=200, verbose=False,
                     normadj=True, setting='trans', reduced=False, reindex=False,
                     **kwargs):

        self.initialize()
        # data for training
        if reduced:
            adj, features, labels, labels_val = to_tensor(data.adj_syn, data.feat_syn, label=data.labels_syn,
                                                          label2=data.labels_val,
                                                          device=self.device)
        elif setting == 'trans':
            adj, features, labels, labels_val = to_tensor(data.adj_full, data.feat_full, data.labels_train,
                                                          data.labels_val, device=self.device)
        else:
            adj, features, labels, labels_val = to_tensor(data.adj_train, data.feat_train, data.labels_train,
                                                          data.labels_val, device=self.device)
        if normadj:
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        if len(data.labels_full.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        elif len(labels.shape) > 1:  # for GCSNTK, use MSE for training
            # print("MSE loss")
            self.float_label = True
            self.loss = torch.nn.MSELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        if reduced or setting == 'ind':
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

        # adj -> adj (SparseTensor)
        adj = dense2sparsetensor(adj)

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

            for batch_size, n_id, adjs in train_loader:
                adjs = [adj[0].to(self.device) for adj in adjs]
                optimizer.zero_grad()
                out = self.forward(features[n_id], adjs)
                loss_train = self.loss(out, labels[n_id[:batch_size]])
                loss_train.backward()
                optimizer.step()

            if verbose and i + 1 % 100 == 0:
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

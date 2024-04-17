from copy import deepcopy

import torch.nn as nn
import torch.optim as optim

from graphslim.models.layers import GraphConvolution
from graphslim.utils import *


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.layers = nn.ModuleList([])
        self.loss = None

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

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):

        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, data, train_iters=200, verbose=False,
                     normalize=True, setting='trans', reduced=False, reindex=False,
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
        if normalize:
            self.adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))
        else:
            self.adj = adj
        self.features = features

        if len(data.labels_full.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        # TODO: we can have two strategies:
        #  1) validate on the original validation set,
        #  2) validate on all the nodes except for test set
        if reduced:
            reindex = True

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        # TODO: we can have two strategies:
        #  1) validate on the original validation set,
        #  2) validate on all the nodes except for test set
        # data for validation
        if setting == 'ind':
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = to_tensor(feat_full, adj_full, device=self.device)
        if normalize:
            adj_full = normalize_adj_tensor(adj_full, sparse=is_sparse_tensor(adj_full))

        self.train()
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr * 0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            optimizer.zero_grad()
            output = self.forward(self.features, self.adj)
            loss_train = self.loss(output if reindex else output[data.idx_train], labels)

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
                    # self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, data=None, verbose=False):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        idx_test = data.idx_test
        # whether condensed or not, use the raw graph to test
        output = self.predict(data.feat_full, data.adj_full)
        # output = self.output
        labels_test = torch.LongTensor(data.labels_test).to(self.device)

        loss_test = F.nll_loss(output[idx_test], labels_test)
        acc_test = accuracy(output[idx_test], labels_test)
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, normadj=True):

        self.eval()
        features, adj = to_tensor(features, adj, device=self.device)
        if normadj:
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        return self.forward(features, adj)

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

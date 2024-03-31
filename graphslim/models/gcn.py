import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

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

    def reset_parameters(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, features, adj, data, labels=None, train_iters=200, initialize=True, verbose=False, normalize=True,
                     patience=None, val=False, **kwargs):
        if initialize:
            self.reset_parameters()

        adj = my_to_tensor(adj, device=self.device) if not isinstance(adj, torch.Tensor) else adj.to(self.device)
        features = my_to_tensor(features, device=self.device) if not isinstance(features,
                                                                                torch.Tensor) else features.to(
            self.device)

        self.adj_norm = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj)) if normalize else adj
        self.features = features

        if len(data.labels_full.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = data.labels_full
        data.labels_full = labels.float() if self.multi_label else labels

        self._train_with_val(data, train_iters, verbose, adj_val=val, **kwargs)

    def _train_with_val(self, data, train_iters, verbose, adj_val=False, condensed=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)
        labels_train = torch.LongTensor(data.labels_syn).to(self.device) if condensed else torch.LongTensor(
            data.labels_train).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr * 0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output if condensed else output[data.idx_train], labels_train)

            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
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
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = to_tensor(features, adj, device=self.device)

            self.features = features
            if is_sparse_tensor(adj):
                self.adj_norm = normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)

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

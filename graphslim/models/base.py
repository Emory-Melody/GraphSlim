from copy import deepcopy

import torch.nn as nn
import torch.optim as optim

from graphslim.utils import *


class BaseGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, args, mode):

        super(BaseGNN, self).__init__()
        self.args = args
        self.with_bn = args.with_bn
        self.with_relu = True
        self.with_bias = True
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.nlayers = args.nlayers
        self.ntrans = args.ntrans
        self.device = args.device
        self.layers = nn.ModuleList([])

        if mode == 'eval':
            self.dropout = 0
            self.weight_decay = 5e-4

        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None
        self.float_label = None

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adj, output_layer_features=False):
        outputs = []

        if isinstance(adj, list):
            for i, layer in enumerate(self.layers):
                x = layer(x, adj[i])
                if i != self.nlayers - 1:
                    x = self.bns[i](x) if self.with_bn else x
                    x = F.relu(x)
                    x = F.dropout(x, self.dropout, training=self.training)
        else:
            for ix, layer in enumerate(self.layers):
                if output_layer_features:
                    outputs.append(x)
                x = layer(x, adj)
                if ix != self.nlayers - 1:
                    x = self.bns[ix](x) if self.with_bn else x
                    if self.with_relu:
                        x = F.relu(x)
                    x = F.dropout(x, self.dropout, training=self.training)

        if output_layer_features:
            outputs.append(x)
            return outputs
        x = x.view(-1, x.shape[-1])
        return F.log_softmax(x, dim=1)


    def fit_with_val(self, data, train_iters=600, verbose=False,
                     normadj=True, setting='trans', reduced=False, reindex=False,
                     **kwargs):

        self.initialize()
        # data for training
        if reduced:
            adj, features, labels, labels_val = to_tensor(data.adj_syn, data.feat_syn, label=data.labels_syn,
                                                          label2=data.labels_val,
                                                          device=self.device)

        elif setting == 'trans':
            adj, features, labels, labels_val = to_tensor(data.adj_full, data.feat_full, label=data.labels_train,
                                                          label2=data.labels_val, device=self.device)
        else:
            adj, features, labels, labels_val = to_tensor(data.adj_train, data.feat_train, label=data.labels_train,
                                                          label2=data.labels_val, device=self.device)
        if self.__class__.__name__ == 'GAT':
            # gat must use SparseTensor
            if len(adj.shape) == 3:
                adj = [normalize_adj_tensor(a.to_sparse(), sparse=True) for a in adj]
            else:
                adj = normalize_adj_tensor(adj.to_sparse(), sparse=True)
        else:
            # others are forced to be dense tensor
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        if self.args.method == 'geom' and self.args.soft_label:
            self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        elif self.args.method == 'gcsntk':  # for GCSNTK, use MSE for training
            self.float_label = True
            self.loss = torch.nn.MSELoss()
        elif len(data.labels_full.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        if reduced or setting == 'ind':
            reindex = True

        if verbose:
            print('=== training ===')

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
        if normadj:
            adj_full = normalize_adj_tensor(adj_full, sparse=is_sparse_tensor(adj_full))

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            if i == train_iters // 2 and self.args.method not in ['geom']:
                optimizer = optim.Adam(self.parameters(), lr=self.lr * 0.1, weight_decay=self.weight_decay)

            self.train()  # ?
            optimizer.zero_grad()
            output = self.forward(features, adj)
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
        return best_acc_val

    def test(self, data=None, setting='trans', verbose=False):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        idx_test = data.idx_test
        # whether condensed or not, use the raw graph to test
        if setting == 'ind':
            output = self.predict(data.feat_test, data.adj_test)
        else:
            output = self.predict(data.feat_full, data.adj_full)

        # output = self.output
        labels_test = torch.LongTensor(data.labels_test).to(self.device)

        if setting == 'ind':
            loss_test = F.nll_loss(output, labels_test)
            acc_test = accuracy(output, labels_test)
        else:
            loss_test = F.nll_loss(output[idx_test], labels_test)
            acc_test = accuracy(output[idx_test], labels_test)

        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, normadj=True, output_layer_features=False):

        self.eval()
        features, adj = to_tensor(features, adj, device=self.device)
        if normadj:
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        return self.forward(features, adj, output_layer_features=output_layer_features)

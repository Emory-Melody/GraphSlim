"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""

import torch.nn.functional as F
from torch import nn

from graphslim.models.gcn import BaseGNN
# from torch_geometric.nn import GATConv
from graphslim.models.layers import GATConv


class GAT(BaseGNN):
    '''
    simple GAT model, one head and no edge weight
    only for evaluation
    '''

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):
        super(GAT, self).__init__(nfeat, nhid, nclass, args, mode)

        if mode in ['eval', 'cross']:
            self.nlayers = 2
            self.weight_decay = 0
            self.lr = 1e-3
            self.heads = 8
            self.output_heads = 1
            nhid //= self.heads
            if args.dataset in ['reddit']:
                self.dropout = 0.1
                self.weight_decay = 5e-4
            elif args.dataset in ['citeseer']:
                self.dropout = 0.7
                self.weight_decay = 5e-4
            elif args.dataset in ['flickr']:
                self.dropout = 0.8
            else:
                self.dropout = 0.7
        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=self.heads,
            dropout=self.dropout,
            bias=self.with_bias)

        self.conv2 = GATConv(
            nhid * self.heads,
            nclass,
            heads=self.output_heads,
            concat=False,
            dropout=self.dropout,
            bias=self.with_bias)

        self.output = None
        self.best_model = None
        self.best_output = None
        self.initialize()

    def forward(self, x, adj, output_layer_features=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

# def test_gat(self, nlayers, model_type, verbose=False):
#     res = []
#     args = self.args
#
#     if args.dataset in ['cora', 'citeseer']:
#         args.epsilon = 0.5  # Make the graph sparser as GAT does not work well on dense graph
#     else:
#         args.epsilon = 0.01
#
#     print('======= testing %s' % model_type)
#     data, device = self.data, self.device
#
#     feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)
#     # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
#     with_bn = False
#     if model_type == 'GAT':
#         model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=16, dropout=0.0,
#                     weight_decay=0e-4, nlayers=self.args.nlayers, lr=0.001,
#                     nclass=data.nclass, device=device, dataset=self.args.dataset).to(device)
#
#     noval = True if args.dataset in ['reddit', 'flickr'] else False
#     model.fit(feat_syn, adj_syn, labels_syn, np.arange(len(feat_syn)), noval=noval, data=data,
#               train_iters=10000 if noval else 3000, normalize=True, verbose=verbose)
#
#     model.eval()
#     labels_test = torch.LongTensor(data.labels_test).to(args.device)
#
#     if args.dataset in ['reddit', 'flickr']:
#         output = model.predict(data.feat_test, data.adj_test)
#         loss_test = F.nll_loss(output, labels_test)
#         acc_test = utils.accuracy(output, labels_test)
#         res.append(acc_test.item())
#         if verbose:
#             print("Test set results:",
#                   "loss= {:.4f}".format(loss_test.item()),
#                   "accuracy= {:.4f}".format(acc_test.item()))
#
#     else:
#         # Full graph
#         output = model.predict(data.feat_full, data.adj_full)
#         loss_test = F.nll_loss(output[data.idx_test], labels_test)
#         acc_test = utils.accuracy(output[data.idx_test], labels_test)
#         res.append(acc_test.item())
#         if verbose:
#             print("Test set results:",
#                   "loss= {:.4f}".format(loss_test.item()),
#                   "accuracy= {:.4f}".format(acc_test.item()))
#
#     labels_train = torch.LongTensor(data.labels_train).to(args.device)
#     output = model.predict(data.feat_train, data.adj_train)
#     loss_train = F.nll_loss(output, labels_train)
#     acc_train = utils.accuracy(output, labels_train)
#     if verbose:
#         print("Train set results:",
#               "loss= {:.4f}".format(loss_train.item()),
#               "accuracy= {:.4f}".format(acc_train.item()))
#     res.append(acc_train.item())
#     return res

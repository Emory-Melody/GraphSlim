from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from graphslim import utils
from graphslim.dataset import *
from graphslim.models import GCN, GAT
from graphslim.utils import accuracy, seed_everything


class Evaluator:

    def __init__(self, args, **kwargs):
        # self.data = data
        self.args = args
        self.device = args.device
        # self.args.runs = 10
        # n = int(data.feat_train.shape[0] * args.reduction_rate)
        # d = data.feat_train.shape[1]
        # self.nnodes_syn = n
        # self.adj_param = nn.Parameter(torch.FloatTensor(n, n).to(device))
        # self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        # self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        # print('adj_param:', self.adj_param.shape, 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        pass
        # self.adj_param.data.copy_(torch.randn(self.adj_param.size()))
        # self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        counter = Counter(data.labels_train.tolist())
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                print(num_class_dict[c])
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_gat(self, nlayers, model_type, verbose=False):
        res = []
        args = self.args

        if args.dataset in ['cora', 'citeseer']:
            args.epsilon = 0.5  # Make the graph sparser as GAT does not work well on dense graph
        else:
            args.epsilon = 0.01

        print('======= testing %s' % model_type)
        data, device = self.data, self.device

        feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)
        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        with_bn = False
        if model_type == 'GAT':
            model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=16, dropout=0.0,
                        weight_decay=0e-4, nlayers=self.args.nlayers, lr=0.001,
                        nclass=data.nclass, device=device, dataset=self.args.dataset).to(device)

        noval = True if args.dataset in ['reddit', 'flickr'] else False
        model.fit(feat_syn, adj_syn, labels_syn, np.arange(len(feat_syn)), noval=noval, data=data,
                  train_iters=10000 if noval else 3000, normalize=True, verbose=verbose)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        if args.dataset in ['reddit', 'flickr']:
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())
        return res

    def get_syn_data(self, data, model_type=None, verbose=False, sparse=False):

        adj_syn, feat_syn, labels_syn = load_reduced(self.args)


        if model_type == 'MLP':
            adj_syn = adj_syn - adj_syn

        if verbose:
            print('Sum:', adj_syn.sum(), adj_syn.sum() / (adj_syn.shape[0] ** 2))
            print('Sparsity:', adj_syn.nonzero().shape[0] / (adj_syn.shape[0] ** 2))

        # Following GCond, when the method is condensation, we use a threshold to sparse the adjacency matrix
        if sparse and self.args.epsilon > 0:
            adj_syn[adj_syn < self.args.epsilon] = 0
            if verbose:
                print('Sparsity after truncating:', adj_syn.nonzero().shape[0] / (adj_syn.shape[0] ** 2))

        # edge_index = adj_syn.nonzero().T
        # adj_syn = torch.sparse.FloatTensor(edge_index,  adj_syn[edge_index[0], edge_index[1]], adj_syn.size())

        return feat_syn.detach(), adj_syn.detach(), labels_syn.detach()

    def test(self, data, model_type, verbose=True):
        args = self.args
        res = []
        feat_syn, adj_syn, labels_syn = data.feat_syn, data.adj_syn, data.labels_syn
        if verbose:
            print('======= testing %s' % model_type)
        if model_type == 'MLP':
            model_class = GCN
        else:
            model_class = eval(model_type)

        model = model_class(nfeat=feat_syn.shape[1], nhid=args.eval_hidden, nclass=data.nclass, nlayers=args.nlayers,
                            dropout=0, lr=args.lr_test, weight_decay=5e-4, device=self.device).to(self.device)

        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        # if self.args.dataset in ['ogbn-arxiv', 'arxiv']:
        #     model = model_class(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.,
        #                         weight_decay=weight_decay, nlayers=self.args.nlayers, with_bn=False,
        #                         nclass=data.nclass, device=self.device).to(self.device)

        model.fit_with_val(data, train_iters=600, normadj=True, normfeat=self.args.normalize_features, verbose=verbose,
                           setting=args.setting,
                           reduced=True)

        model.eval()
        labels_test = data.labels_test.long().cuda()

        # if model_type == 'MLP':
        #     output = model.predict(data.feat_test, sp.eye(len(data.feat_test),normadj))
        output = model.predict(data.feat_test, data.adj_test)

        if self.args.setting == 'ind':
            loss_test = F.nll_loss(output, labels_test)
            acc_test = accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        # if not args.dataset in ['reddit', 'flickr']:
        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test full set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

            # labels_train = torch.LongTensor(data.labels_train).cuda()
            # output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = accuracy(output, labels_train)
            # if verbose:
            #     print("Train set results:",
            #           "loss= {:.4f}".format(loss_train.item()),
            #           "accuracy= {:.4f}".format(acc_train.item()))
            # res.append(acc_train.item())
        return res

    def train_cross(self, verbose=True):
        args = self.args
        data = self.data
        data.nclass = data.nclass.item()

        final_res = {}

        for model_type in ['GCN', 'GraphSage', 'SGCRich', 'MLP', 'APPNPRich', 'Cheby']:
            res = []
            for i in range(args.runs):
                res.append(self.test(model_type=model_type, verbose=False))
            res = np.array(res)
            print('Test/Train Mean Accuracy:',
                  repr([res.mean(0), res.std(0)]))
            final_res[model_type] = [res.mean(0), res.std(0)]

        # print('=== testing GAT')
        # res = []
        # nlayer = 2
        # for i in range(runs):
        #     res.append(self.test_gat(verbose=True, nlayers=nlayer, model_type='GAT'))
        # res = np.array(res)
        # print('Layer:', nlayer)
        # print('Test/Full Test/Train Mean Accuracy:',
        #         repr([res.mean(0), res.std(0)]))
        # final_res['GAT'] = [res.mean(0), res.std(0)]

        print('Final result:', final_res)

    def evaluate(self, data, model_type, verbose=True):
        # model_type: ['GCN1', 'GraphSage', 'SGC1', 'MLP', 'APPNP1', 'Cheby']
        # self.data = data
        args = self.args

        res = []
        data.feat_syn, data.adj_syn, data.labels_syn = self.get_syn_data(data, model_type)
        for i in trange(args.runs):
            seed_everything(args.seed + i)
            res.append(self.test(data, model_type=model_type, verbose=False))
        res = np.array(res)

        print(f'Test Mean Accuracy: {100 * res.mean(0)[0]:.2f} +/- {100 * res.std(0)[0]:.2f}')
        return res
        # else:
        #     return res[0][0]
        # print('Test/Train Mean Accuracy:',
        #       repr([res.mean(0), res.std(0)]))
        # final_res[model_type] = [res.mean(0), res.std(0)]
        #
        # print('Final result:', final_res)

from collections import Counter

import torch.nn as nn

from graphslim.condensation.utils import *
from graphslim.evaluation import Evaluator
from graphslim.models import *
from graphslim.utils import *


class GCondBase:

    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device
        self.setting = setting

        # n = data.nclass * args.nsamples

        self.data.labels_syn = self.generate_labels_syn(data)
        n = self.data.labels_syn.shape[0]
        self.nnodes_syn = n
        d = data.feat_train.shape[1]
        self.d = d
        print(f'target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}')
        print(f'actual reduced size:{n}')

        # from collections import Counter; print(Counter(data.labels_train))

        self.feat_syn = nn.Parameter(torch.empty(n, d).to(self.device))
        self.pge = PGE(nfeat=d, nnodes=n, device=self.device, args=args).to(self.device)
        self.adj_syn = None

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.pge.reset_parameters()

    def generate_labels_syn(self, data):
        counter = Counter(data.labels_train.tolist())
        num_class_dict = {}
        # n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return np.array(labels_syn)

    def cross_architecture_eval(self):
        args = self.args
        data = self.data

        if args.dataset in ['cora', 'citeseer']:
            args.epsilon = 0.05
        else:
            args.epsilon = 0.01

        agent = Evaluator(data, args, device='cuda')
        agent.train_cross()

    def train_class(self, model, adj, features, labels, labels_syn, args):
        data = self.data
        feat_syn = self.feat_syn
        adj_syn_norm = self.adj_syn
        syn_class_indices = self.syn_class_indices
        model_parameters = list(model.parameters())
        loss = torch.tensor(0.0).to(self.device)
        for c in range(data.nclass):
            batch_size, n_id, adjs = data.retrieve_class_sampler(
                c, adj, args)
            if args.nlayers == 1:
                adjs = [adjs]

            adjs = [adj.to(self.device) for adj in adjs]
            output = model.forward_sampler(features[n_id], adjs)
            loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
            gw_real = torch.autograd.grad(loss_real, model_parameters)
            gw_real = list((_.detach().clone() for _ in gw_real))
            if args.setting == 'ind':
                ind = syn_class_indices[c]
                if args.nlayers == 1:
                    adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                else:
                    adj_syn_norm_list = [adj_syn_norm] * (args.nlayers - 1) + \
                                        [adj_syn_norm[ind[0]: ind[1]]]

                output_syn = model.forward_syn(feat_syn, adj_syn_norm_list)
                loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])
            else:
                output_syn = model.forward(feat_syn, adj_syn_norm)
                ind = syn_class_indices[c]
                loss_syn = F.nll_loss(
                    output_syn[ind[0]: ind[1]],
                    labels_syn[ind[0]: ind[1]])

            gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
            coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
            loss += coeff * match_loss(gw_syn, gw_real, args, device=self.device)

        return loss

    def get_sub_adj_feat(self):
        data = self.data
        args = self.args
        idx_selected = []

        counter = Counter(self.data.labels_syn)

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = data.feat_train[idx_selected]
        args.knnsamples = 3
        adj_knn = torch.zeros((self.nnodes_syn, self.nnodes_syn)).to(self.device)

        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        # from sklearn.metrics.pairwise import cosine_similarity
        # # features[features!=0] = 1
        # k = 2
        # sims = cosine_similarity(features.cpu().numpy())
        # sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        # for i in range(len(sims)):
        #     indices_argsort = np.argsort(sims[i])
        #     sims[i, indices_argsort[: -k]] = 0
        # adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn

    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.

        if args.method in ['doscond', 'doscondx']:
            if args.dataset == 'ogbn-arxiv':
                return 5, 0
            return 1, 0
        if args.method == 'sgdd':
            if args.dataset in ['ogbn-arxiv']:
                return 20, 3
            if args.dataset in ['cora']:
                return 20, 15
            if args.dataset in ['citeseer']:
                return 20, 15
            if args.dataset in ['physics']:
                return 20, 10
            else:
                return 20, 10
        if args.method in ['gcond', 'gcondx']:
            if args.dataset in ['flickr', 'reddit']:
                return 10, 1
            if args.dataset in ['ogbn-arxiv', 'cora', 'citeseer']:
                return 20, 15
        return 10, 1

    def check_bn(self, model):
        BN_flag = False
        for module in model.modules():
            if 'BatchNorm' in module._get_name():  # BatchNorm
                BN_flag = True
        if BN_flag:
            model.train()  # for updating the mu, sigma of BatchNorm
            # output_real = model.forward(features, adj)
            for module in model.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    module.eval()  # fix mu and sigma of every BatchNorm layer
        return model

    def intermediate_evaluation(self):
        pass

    def test_with_val(self, verbose=False, setting='trans'):
        res = []

        args, data, device = self.args, self.data, self.device

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN(nfeat=data.feat_syn.shape[1], nhid=args.hidden, dropout=args.dropout,
                    weight_decay=args.weight_decay, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        # if self.args.lr_adj == 0:
        #     n = len(data.labels_syn)
        #     adj_syn = torch.zeros((n, n))
        # same for ind and trans when reduced
        acc_val = model.fit_with_val(data,
                                     train_iters=600, normadj=True, verbose=False,
                                     setting=setting, reduced=True)
        # model.eval()
        # labels_test = data.labels_test.long().to(args.device)
        # if setting == 'trans':
        #
        #     output = model.predict(data.feat_full, data.adj_full)
        #     acc_test = accuracy(output[data.idx_test], labels_test)
        #
        # else:
        #     output = model.predict(data.feat_test, data.adj_test)
        #     # loss_test = F.nll_loss(output, labels_test)
        #     acc_test = accuracy(output, labels_test)
        # res.append(acc_test.item())
        res.append(acc_val.item())
        # if verbose:
        #     print('Val Accuracy and Std:',
        #           repr([res.mean(0), res.std(0)]))
        return res

        # print(adj_syn.sum(), adj_syn.sum() / (adj_syn.shape[0] ** 2))

        # if False:
        #     if self.args.dataset == 'ogbn-arxiv':
        #         thresh = 0.6
        #     elif self.args.dataset == 'reddit':
        #         thresh = 0.91
        #     else:
        #         thresh = 0.7
        #
        #     labels_train = torch.LongTensor(data.labels_train).cuda()
        #     output = model.predict(data.feat_train, data.adj_train)
        #     # loss_train = F.nll_loss(output, labels_train)
        #     # acc_train = utils.accuracy(output, labels_train)
        #     loss_train = torch.tensor(0)
        #     acc_train = torch.tensor(0)
        #     if verbose:
        #         print("Train set results:",
        #               "loss= {:.4f}".format(loss_train.item()),
        #               "accuracy= {:.4f}".format(acc_train.item()))
        #     res.append(acc_train.item())

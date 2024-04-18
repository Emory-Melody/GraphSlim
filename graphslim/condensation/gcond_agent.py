from collections import Counter

import torch.nn as nn

from graphslim.condensation.utils import match_loss  # graphslim
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import Evaluator
from graphslim.models import GCN, PGE, SGC1, SGC
from graphslim.utils import *


class GCond:

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

    def reduce(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = to_tensor(self.feat_syn, self.pge, data.labels_syn, device=self.device)
        features, adj, labels = to_tensor(data.feat_full, data.adj_full, data.labels_full, device=self.device)
        # idx_train = data.idx_train

        syn_class_indices = self.syn_class_indices

        # initialization the features
        feat_sub, adj_sub = self.get_sub_adj_feat()
        self.feat_syn.data.copy_(feat_sub)

        adj = normalize_adj_tensor(adj, sparse=True)
        # adj = SparseTensor(row=adj.indices()[0], col=adj.indices()[1],
        #                    value=adj.values(), sparse_sizes=adj.size()).t()
        #
        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        # best_val = 0

        for it in range(args.epochs):
            # seed_everything(args.seed + it)
            if args.dataset in ['ogbn-arxiv']:
                model = SGC1(nfeat=feat_syn.shape[1], nhid=args.hidden,
                             dropout=0.0, with_bn=False,
                             weight_decay=0e-4, nlayers=2,
                             nclass=data.nclass,
                             device=self.device).to(self.device)
            else:
                model = SGC(nfeat=feat_syn.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=0, weight_decay=args.weight_decay,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)
                # feat_syn_norm = feat_syn

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
                    output_syn = model.forward(feat_syn, adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                        output_syn[ind[0]: ind[1]],
                        labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # if args.alpha > 0:
                #     loss_reg = args.alpha * regularization(adj_syn, tensor2onehot(labels_syn))
                # else:
                # loss_reg = torch.tensor(0)

                # loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if args.one_step:
                    self.optimizer_feat.step()
                    self.optimizer_pge.step()
                else:
                    if it % 50 < 10:
                        self.optimizer_pge.step()
                    else:
                        self.optimizer_feat.step()
                # else:
                #     if ol < outer_loop // 5:
                #         self.optimizer_pge.step()
                #     else:
                #         self.optimizer_feat.step()

                # if verbose and ol % 5 == 0:
                #     print('Gradient matching loss:', loss.item())

                # if ol == outer_loop - 1:
                #     # print('loss_reg:', loss_reg.item())
                #     # print('Gradient matching loss:', loss.item())
                #     break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)
                if args.normalize_features:
                    feat_syn_inner_norm = F.normalize(feat_syn_inner, dim=0)
                else:
                    feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step()  # update gnn param

            loss_avg /= (data.nclass * outer_loop)
            if verbose and (it + 1) % 100 == 0:
                print('Epoch {}, loss_avg: {}'.format(it + 1, loss_avg))

            # eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            eval_epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            # if it == 0:
            data.adj_syn, data.feat_syn, data.labels_syn = adj_syn_inner.detach(), feat_syn_inner.detach(), labels_syn.detach()

            if verbose and it + 1 in eval_epochs:
                # if verbose and (it+1) % 50 == 0:
                res = []
                for i in range(3):
                    res.append(self.test_with_val(verbose=False, setting=args.setting))

                res = np.array(res)
                current_val = res.mean()
                print('Test Accuracy and Std:',
                      repr([current_val, res.std()]))

                # if current_val > best_val:
                #     best_val = current_val
                #     data.adj_syn, data.feat_syn, data.labels_syn = adj_syn_inner.detach(), feat_syn_inner.detach(), labels_syn.detach()

        if args.save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        return data

    def cross_architecture_eval(self):
        args = self.args
        data = self.data

        if args.dataset in ['cora', 'citeseer']:
            args.epsilon = 0.05
        else:
            args.epsilon = 0.01

        agent = Evaluator(data, args, device='cuda')
        agent.train_cross()

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
        # TODO: fix the bug, there is no "one_step" in args (AttributeError: 'Obj' object has no attribute 'one_step')
        # if args.one_step:
        #     return 10, 0
        if args.one_step:
            if args.dataset == 'ogbn-arxiv':
                return 5, 0
            return 1, 0
        if args.dataset in ['ogbn-arxiv']:
            return 20, 0
        if args.dataset in ['reddit']:
            return 10, 1
        if args.dataset in ['flickr']:
            return 10, 1
            # return 10, 1
        if args.dataset in ['cora']:
            return 20, 1
        if args.dataset in ['citeseer']:
            return 20, 5  # at least 200 epochs
        else:
            return 20, 1

    def test_with_val(self, verbose=True, setting='trans'):
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
                                     train_iters=600, normadj=True, normfeat=args.normalize_features, verbose=False,
                                     reduced=True)
        # model.eval()
        labels_test = data.labels_test.long().to(args.device)
        if setting == 'trans':

            output = model.predict(data.feat_full, data.adj_full)
            acc_test = accuracy(output[data.idx_test], labels_test)

        else:
            output = model.predict(data.feat_test, data.adj_test)
            # loss_test = F.nll_loss(output, labels_test)
            acc_test = accuracy(output, labels_test)
        res.append(acc_test.item())
        # res.append(acc_val.item())
        if verbose:
            print('Test Accuracy and Std:',
                  repr([res.mean(0), res.std(0)]))
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

from collections import Counter

import torch.nn as nn

from graphslim.coarsening import *
from graphslim.condensation.utils import *
from graphslim.models import *
from graphslim.sparsification import *
from graphslim.utils import *
from graphslim.dataset.utils import save_reduced


class GCondBase:

    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        args.epochs -= 1
        self.device = args.device
        self.setting = setting

        if args.method not in ['msgc']:
            self.labels_syn = self.data.labels_syn = self.generate_labels_syn(data)
            n = self.nnodes_syn = self.data.labels_syn.shape[0]
        else:
            n = self.nnodes_syn = int(data.feat_train.shape[0] * args.reduction_rate)
        self.d = d = data.feat_train.shape[1]
        # self.d = d = 64
        print(f'target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}')
        print(f'actual reduced size:{n}')

        # from collections import Counter; print(Counter(data.labels_train))

        self.feat_syn = nn.Parameter(torch.empty(n, d).to(self.device))
        if args.method not in ['sgdd', 'gcsntk', 'msgc']:
            self.pge = PGE(nfeat=d, nnodes=n, device=self.device, args=args).to(self.device)
            self.adj_syn = None

            # self.reset_parameters()
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
            self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
            print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        self.pge.reset_parameters()

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
                # only clip labels with largest number of samples
                num_class_dict[c] = max(int(n * self.args.reduction_rate) - sum_, 1)
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
        self.data.num_class_dict = self.num_class_dict = num_class_dict
        if self.args.verbose:
            print(num_class_dict)
        return np.array(labels_syn)

    def init(self, with_adj=False):
        args = self.args
        if args.init == 'clustering':
            agent = Cluster(setting=args.setting, data=self.data, args=args)
        elif args.init == 'averaging':
            agent = Average(setting=args.setting, data=self.data, args=args)
        elif args.init == 'kcenter':
            agent = KCenter(setting=args.setting, data=self.data, args=args)
        elif args.init == 'herding':
            agent = Herding(setting=args.setting, data=self.data, args=args)
        elif args.init == 'cent_p':
            agent = CentP(setting=args.setting, data=self.data, args=args)
        elif args.init == 'cent_d':
            agent = CentD(setting=args.setting, data=self.data, args=args)
        else:
            agent = Random(setting=args.setting, data=self.data, args=args)

        reduced_data = agent.reduce(self.data, verbose=True, save=False)
        if with_adj:
            return reduced_data.feat_syn, reduced_data.adj_syn
        else:
            return reduced_data.feat_syn

    def train_class(self, model, adj, features, labels, labels_syn, args):
        data = self.data
        feat_syn = self.feat_syn
        adj_syn_norm = self.adj_syn
        loss = torch.tensor(0.0).to(self.device)
        for c in range(data.nclass):
            batch_size, n_id, adjs = data.retrieve_class_sampler(
                c, adj, args)
            adjs = [adj[0].to(self.device) for adj in adjs]
            output = model(features[n_id], adjs)
            loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
            gw_reals = torch.autograd.grad(loss_real, model.parameters())
            gw_reals = list((_.detach().clone() for _ in gw_reals))
            # ------------------------------------------------------------------
            output_syn = model(feat_syn, adj_syn_norm)
            loss_syn = F.nll_loss(output_syn[labels_syn == c], labels_syn[labels_syn == c])
            gw_syns = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
            # ------------------------------------------------------------------
            coeff = self.num_class_dict[c] / self.nnodes_syn
            ml = match_loss(gw_syns, gw_reals, args, device=self.device)
            loss += coeff * ml

        return loss

    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.

        return args.outer_loop, args.inner_loop

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

    def intermediate_evaluation(self, best_val, loss_avg, save=True):
        data = self.data
        args = self.args
        if args.verbose:
            print('loss_avg: {}'.format(loss_avg))

        res = []

        for i in range(args.run_inter_eval):
            # small epochs for fast intermediate evaluation
            res.append(self.test_with_val(verbose=False, setting=args.setting, iters=args.eval_epochs))

        res = np.array(res)
        current_val = res.mean()
        args.logger.info('Val Accuracy and Std:' + repr([current_val, res.std()]))

        if save and current_val > best_val:
            best_val = current_val
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        return best_val

    def test_with_val(self, verbose=False, setting='trans', iters=200):
        res = []

        args, data, device = self.args, self.data, self.device

        model = GCN(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='eval').to(device)

        acc_val = model.fit_with_val(data,
                                     train_iters=iters, normadj=True, verbose=False,
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

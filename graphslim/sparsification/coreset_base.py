import time
from collections import Counter

import numpy as np
import torch
from torch_sparse import matmul

from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from graphslim.utils import normalize_adj_tensor
from graphslim.utils import to_tensor, getsize_mb


class CoreSet:

    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.setting = setting

        self.device = args.device
        self.num_class_dict, self.labels_train, self.idx_train = self.prepare_select(data, args)
        # n = int(data.feat_train.shape[0] * args.reduction_rate)

    def reduce(self, data, verbose=False):
        if verbose:
            start = time.perf_counter()
        args = self.args
        if self.setting == 'trans':
            if args.aggpreprocess:
                data.adj_fully = to_tensor(data.adj_full)[0]
                data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
                data.pre_conv = matmul(data.pre_conv, data.pre_conv)
                data.feat_syn = matmul(data.pre_conv, data.feat_full)[idx_selected].float()
                data.adj_syn = torch.eye(data.feat_syn.shape[0], device=args.device)
                data.labels_syn = data.labels_full[idx_selected]

            else:
                model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                            weight_decay=args.weight_decay).to(args.device)
                model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose, setting='trans')
                # model.test(data, verbose=True)
                embeds = model.predict(data.feat_full, data.adj_full).detach()

                idx_selected = self.select(embeds)

                # induce a graph with selected nodes

                data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]
                data.feat_syn = data.feat_full[idx_selected]
                data.labels_syn = data.labels_full[idx_selected]

        if self.setting == 'ind':
            if args.aggpreprocess:
                data.adj_fully = to_tensor(data.adj_train)[0]
                data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
                data.pre_conv = matmul(data.pre_conv, data.pre_conv)
                data.feat_syn = matmul(data.pre_conv, data.feat_train)[idx_selected].float()
                data.adj_syn = torch.eye(data.feat_syn.shape[0], device=args.device)
                data.labels_syn = data.labels_train[idx_selected]
            else:
                model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                            weight_decay=args.weight_decay).to(args.device)
                model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose, setting='ind', reindex=True)

                model.eval()

                embeds = model.predict(data.feat_train, data.adj_train).detach()

                idx_selected = self.select(embeds)
                data.feat_syn = data.feat_train[idx_selected]
                data.adj_syn = data.adj_train[np.ix_(idx_selected, idx_selected)]
                data.labels_syn = data.labels_train[idx_selected]

        print('selected nodes:', idx_selected.shape[0])
        print('induced edges:', data.adj_syn.sum())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn,
                                                                 device='cpu')
        save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        if verbose:
            end = time.perf_counter()
            runTime = end - start
            runTime_ms = runTime * 1000
            print("Reduce Time: ", runTime, "s")
            print("Reduce Time: ", runTime_ms, "ms")
            if args.setting == 'trans':
                origin_storage = getsize_mb([data.x, data.edge_index, data.y])
            else:
                origin_storage = getsize_mb([data.feat_train, data.adj_train, data.labels_train])
            condensed_storage = getsize_mb([data.feat_syn, data.adj_syn, data.labels_syn])
            print(f'Origin graph:{origin_storage:.2f}Mb  Condensed graph:{condensed_storage:.2f}Mb')

        return data

    def prepare_select(self, data, args):
        num_class_dict = {}
        syn_class_indices = {}
        if args.setting == 'ind':
            idx_train = np.arange(len(data.idx_train))
        else:
            idx_train = data.idx_train
        labels_train = data.labels_train
        # d = data.feat_train.shape[1]
        counter = Counter(data.labels_train.tolist())
        # n = len(data.labels_train)
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        return num_class_dict, labels_train, idx_train

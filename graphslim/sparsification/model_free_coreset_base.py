import time

import numpy as np
import torch
from torch_sparse import matmul

from graphslim.dataset.utils import save_reduced
from graphslim.sparsification.coreset_base import CoreSet
from graphslim.utils import normalize_adj_tensor
from graphslim.utils import to_tensor, getsize_mb


class MFCoreSet(CoreSet):

    def reduce(self, data, verbose=False):
        if verbose:
            start = time.perf_counter()
        args = self.args
        if self.setting == 'trans':
            if args.aggpreprocess:
                data.adj_fully = to_tensor(data.adj_full)[0]
                data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
                data.pre_conv = matmul(data.pre_conv, data.pre_conv)
                feat_agg = matmul(data.pre_conv, data.feat_full).float()
                idx_selected = self.select(feat_agg)

                data.feat_syn = feat_agg[idx_selected]
                data.adj_syn = torch.eye(data.feat_syn.shape[0], device=args.device)
                data.labels_syn = data.labels_full[idx_selected]
            else:
                idx_selected = self.select()
                data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]
                data.feat_syn = data.feat_full[idx_selected]
                data.labels_syn = data.labels_full[idx_selected]

        if self.setting == 'ind':
            if args.aggpreprocess:
                data.adj_fully = to_tensor(data.adj_train)[0]
                data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
                data.pre_conv = matmul(data.pre_conv, data.pre_conv)
                feat_agg = matmul(data.pre_conv, data.feat_full).float()
                idx_selected = self.select(feat_agg)

                data.feat_syn = feat_agg[idx_selected]
                data.adj_syn = torch.eye(data.feat_syn.shape[0], device=args.device)
                data.labels_syn = data.labels_train[idx_selected]
            else:
                idx_selected = self.select()
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

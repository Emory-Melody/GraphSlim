import numpy as np
import torch
from torch_sparse import matmul

from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.sparsification.coreset_base import CoreSet
from graphslim.utils import normalize_adj_tensor, to_tensor


class MFCoreSet(CoreSet):
    def __init__(self, setting, data, args, **kwargs):
        super(MFCoreSet, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=False, save=True):

        args = self.args
        if self.setting == 'trans':
            if args.aggpreprocess:
                data.adj_fully = to_tensor(data.adj_full)
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
                data.adj_fully = to_tensor(data.adj_train)
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

        if verbose:
            print('selected nodes:', idx_selected.shape[0])
            print('induced edges:', data.adj_syn.sum())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(data.adj_syn, data.feat_syn, label=data.labels_syn,
                                                                 device='cpu')
        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        return data

from collections import Counter

import numpy as np
import torch
from sklearn.cluster import BisectingKMeans
from torch_scatter import scatter_mean

from graphslim.coarsening.coarsening_base import Coarsen
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory

from torch_sparse import matmul
from graphslim.utils import normalize_adj_tensor, to_tensor


class ClusterAgg(Coarsen):
    # a structure free coarsening method
    # also serve as initialization for condensation methods
    # output: feat_syn, label_syn
    def __init__(self, setting, data, args, **kwargs):
        super(ClusterAgg, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True, save=True):
        args = self.args
        n_classes = data.nclass
        if args.setting == 'trans':
            data.adj_fully = to_tensor(data.adj_full)
            data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
            data.pre_conv = matmul(data.pre_conv, data.pre_conv)
            feat_agg = matmul(data.pre_conv, data.feat_full).float()
        else:
            data.adj_fully = to_tensor(data.adj_train)
            data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
            data.pre_conv = matmul(data.pre_conv, data.pre_conv)
            feat_agg = matmul(data.pre_conv, data.feat_train).float()

        if hasattr(data, 'labels_syn') and data.labels_syn is not None:
            y_syn = data.labels_syn
            self.labels_train = data.labels_train
            y_train = data.labels_train
        else:
            y_syn, y_train = self.prepare_select(data, args)
        x_train = feat_agg[data.train_mask] if args.setting == 'trans' else feat_agg
        x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1])
        for c in range(n_classes):
            x_c = x_train[y_train == c].cpu()
            n_c = (y_syn == c).sum().item()
            k_means = BisectingKMeans(n_clusters=n_c, random_state=0)
            k_means.fit(x_c)
            clusters = torch.from_numpy(k_means.predict(x_c)).long()
            x_syn[y_syn == c] = scatter_mean(x_c, clusters, dim=0)
        data.feat_syn, data.labels_syn = x_syn.to(x_train.device), y_syn
        data.adj_syn = torch.eye(data.feat_syn.shape[0])
        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        return data

    def prepare_select(self, data, args):
        num_class_dict = {}
        syn_class_indices = {}

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
        labels_syn = np.array(labels_syn)

        return labels_syn, labels_train

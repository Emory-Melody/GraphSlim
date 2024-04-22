from collections import Counter

import numpy as np
import torch
from sklearn.cluster import BisectingKMeans
from torch_scatter import scatter_mean

from graphslim.coarsening.coarsening_base import Coarsen


class Cluster(Coarsen):
    def initialize_x_syn(self, n_classes):
        data = self.data
        args = self.args
        y_syn, y_train, x_train = self.prepare_select(data, args)
        x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1])
        for c in range(n_classes):
            x_c = x_train[y_train == c].cpu()
            n_c = (y_syn == c).sum().item()
            k_means = BisectingKMeans(n_clusters=n_c, random_state=0)
            k_means.fit(x_c)
            clusters = torch.LongTensor(k_means.predict(x_c))
            x_syn[y_syn == c] = scatter_mean(x_c, clusters, dim=0)
        return x_syn.to(x_train.device)

    def prepare_select(self, data, args):
        num_class_dict = {}
        syn_class_indices = {}
        if args.setting == 'ind':
            # idx_train = np.arange(len(data.idx_train))
            feat_train = data.feat_train
        else:
            # idx_train = data.idx_train
            feat_train = data.feat_full
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

        return labels_syn, labels_train, feat_train

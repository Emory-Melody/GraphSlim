from collections import Counter

import numpy as np


class CoreSet:

    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.setting = setting

        self.device = args.device
        if hasattr(data, 'labels_syn') and data.labels_syn is not None:
            self.num_class_dict = data.num_class_dict
            self.labels_train = data.labels_train
            if args.setting == 'ind':
                self.idx_train = np.arange(len(data.idx_train))
            else:
                self.idx_train = data.idx_train
        else:
            self.num_class_dict, self.labels_train, self.idx_train = self.prepare_select(data, args)
        self.condense_model = 'GCN'
        # n = int(data.feat_train.shape[0] * args.reduction_rate)

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

# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
from collections import Counter

import numpy as np


class CoreSet:
    """Base for classwise node selection and per-class synthetic node budgets."""

    def __init__(self, setting, data, args, **kwargs):
        """Initialize CoreSet and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        self.data = data
        self.args = args
        self.setting = setting

        self.device = args.device
        if hasattr(data, "labels_syn") and data.labels_syn is not None:
            self.num_class_dict = data.num_class_dict
            self.labels_train = data.labels_train
            if args.setting == "ind":
                self.idx_train = np.arange(len(data.idx_train))
            else:
                self.idx_train = data.idx_train
        else:
            self.num_class_dict, self.labels_train, self.idx_train = self.prepare_select(data, args)
        self.condense_model = "GCN"
        # n = int(data.feat_train.shape[0] * args.reduction_rate)

    def prepare_select(self, data, args):
        """Allocate at least one representative per observed training class.

        Each class budget is ``max(int(class_size * reduction_rate), 1)``.

        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :returns: Tuple ``(class_budget, training_labels, training_node_ids)``.
        """
        num_class_dict = {}
        syn_class_indices = {}
        if args.setting == "ind":
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

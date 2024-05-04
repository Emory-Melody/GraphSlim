import numpy as np

from graphslim.sparsification.model_free_coreset_base import MFCoreSet


class Random(MFCoreSet):
    def __init__(self, setting, data, args, **kwargs):
        super(Random, self).__init__(setting, data, args, **kwargs)
    def select(self, embedds=None):
        idx_selected = []

        for class_id, cnt in self.num_class_dict.items():
            idx = self.idx_train[self.labels_train == class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        return np.hstack(idx_selected)

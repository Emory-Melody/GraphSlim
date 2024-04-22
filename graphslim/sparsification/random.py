import numpy as np

from graphslim.sparsification.coreset_base import CoreSet


class Random(CoreSet):
    def select(self, embedds):
        idx_selected = []

        for class_id, cnt in self.num_class_dict.items():
            idx = self.idx_train[self.labels_train == class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        return np.hstack(idx_selected)

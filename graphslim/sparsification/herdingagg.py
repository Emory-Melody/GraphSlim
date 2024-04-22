import numpy as np
import torch

from graphslim.sparsification.model_free_coreset_base import MFCoreSet


class HerdingAgg(MFCoreSet):
    def select(self, embeds):

        idx_selected = []
        for class_id, cnt in self.num_class_dict.items():
            idx = self.idx_train[self.labels_train == class_id]
            features = embeds[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                det = mean * (i + 1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                selected.append(idx_left[id_min])
                del idx_left[id_min]
            idx_selected.append(idx[selected])
        return np.hstack(idx_selected)

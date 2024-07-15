import numpy as np
import torch

from graphslim.sparsification.model_free_coreset_base import MFCoreSet


class KCenterAgg(MFCoreSet):

    def select(self, embeds):
        idx_selected = []
        for class_id, cnt in self.num_class_dict.items():
            idx = self.idx_train[self.labels_train == class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            dis = torch.cdist(feature, mean)[:, 0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for i in range(cnt - 1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)

            idx_selected.append(idx[idx_centers])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)

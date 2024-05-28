from graphslim.sparsification.model_based_coreset_base import MBCoreSet
import torch
import numpy as np
from collections import Counter
import random


class KCenterSample(MBCoreSet):

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
#
#   def get_sub_adj_feat(self, features):
#       data = self.data
#       args = self.args
#       idx_selected = []
#
#       counter = Counter(self.data.labels_syn)
#       labels_train = self.data.labels_train.squeeze().tolist()  # important
#       ids_per_cls_train = [(labels_train == c).nonzero()[0] for c in counter.keys()]
#       idx_selected = self.sampling(ids_per_cls_train, counter, features, 0.5, counter)
#       features = features[idx_selected]
#
#       return features, None
#
# def sampling(self, ids_per_cls_train, budget, vecs, d):
#       budget_dist_compute = 1000
#       '''
#       if using_half:
#           vecs = vecs.half()
#       '''
#       if isinstance(vecs, np.ndarray):
#           vecs = torch.from_numpy(vecs)
#       vecs = vecs.half()
#       ids_selected = []
#       for i, ids in enumerate(ids_per_cls_train):
#           class_ = list(budget.keys())[i]
#           other_cls_ids = list(range(len(ids_per_cls_train)))
#           other_cls_ids.pop(i)
#           ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(
#               ids_per_cls_train[i], k=budget_dist_compute)
#
#           dist = []
#           vecs_0 = vecs[ids_selected0]
#           for j in other_cls_ids:
#               chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute, len(ids_per_cls_train[j])))
#               vecs_1 = vecs[chosen_ids]
#               if len(chosen_ids) < 26 or len(ids_selected0) < 26:
#                   # torch.cdist throws error for tensor smaller than 26
#                   dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
#               else:
#                   dist.append(torch.cdist(vecs_0, vecs_1))
#
#           # dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
#           dist_ = torch.cat(dist, dim=-1)  # include distance to all the other classes
#           n_selected = (dist_ < d).sum(dim=-1)
#           rank = n_selected.sort()[1].tolist()
#           current_ids_selected = rank[:budget[class_]] if len(rank) > budget[class_] else random.choices(rank,
#                                                                                                          k=budget[
#                                                                                                              class_])
#           ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
#       return ids_selected

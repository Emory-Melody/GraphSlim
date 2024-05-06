import numpy as np

from graphslim.sparsification.model_free_coreset_base import MFCoreSet


class CentD(MFCoreSet):
    # select nodes with topk PR value in each class
    def select(self, embedds=None):
        if self.args.setting == 'ind':
            adj = self.data.adj_train.astype(np.uint8).todense()
        else:
            adj = self.data.adj_full.astype(np.uint8).todense()
        pr = np.asarray(np.sum(adj, axis=1)).ravel()
        idx_selected = []

        for class_id, cnt in self.num_class_dict.items():
            # Get indices of nodes in the training set that belong to the current class
            idx = self.idx_train[self.labels_train == class_id]

            pr_values = pr[idx]

            topk_indices = np.argsort(pr_values)[-cnt:]
            selected = idx[topk_indices]

            idx_selected.append(selected)

        # Concatenate all selected indices into a single array
        return np.hstack(idx_selected)

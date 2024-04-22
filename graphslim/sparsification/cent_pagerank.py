import numpy as np

from graphslim.sparsification.model_free_coreset_base import MFCoreSet


class CentP(MFCoreSet):
    # select nodes with topk PR value in each class
    def select(self, embedds=None):
        pr = self.pagerank_algorithm()  # Retrieve PageRank values, assumed to be a dictionary or array
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

    def pagerank_algorithm(self, damping_factor=0.85, max_iterations=100, convergence_threshold=0.0001):
        if self.args.setting == 'ind':
            adj = self.data.adj_train.todense()
        else:
            adj = self.data.adj_full.todense()
        n = adj.shape[0]

        # 构建转移矩阵
        transition_matrix = adj / np.sum(adj, axis=0)

        # 初始化PageRank向量
        pagerank = (np.ones(n) / n).reshape(-1, 1)

        # 开始迭代
        for i in range(max_iterations):
            old_pagerank = np.copy(pagerank)

            # 计算新的PageRank向量
            pagerank = damping_factor * np.dot(transition_matrix, old_pagerank) + (1 - damping_factor) / n

            # 判断是否收敛
            if np.sum(np.abs(pagerank - old_pagerank)) < convergence_threshold:
                break

        return np.asarray(pagerank).ravel()

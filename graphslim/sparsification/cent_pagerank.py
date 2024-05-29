import numpy as np

from graphslim.sparsification.model_free_coreset_base import MFCoreSet
from scipy.sparse import csr_matrix, diags
from numpy.linalg import norm


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
            adj = self.data.adj_train.astype(np.uint8)
        else:
            adj = self.data.adj_full.astype(np.uint8)

        n = adj.shape[0]
        adj = csr_matrix(adj)

        # Calculate out-degree
        out_degree = np.array(adj.sum(axis=1)).flatten()
        out_degree[out_degree == 0] = 1  # Avoid division by zero for isolated nodes

        # Create transition matrix
        transition_matrix = adj.multiply(1.0 / out_degree[:, None])

        # Initialize PageRank vector
        pagerank = np.ones((n, 1)) / n
        momentum = (1 - damping_factor) * np.ones((n, 1)) / n

        # Iterate to compute PageRank
        for i in range(max_iterations):
            old_pagerank = pagerank.copy()
            pagerank = damping_factor * (transition_matrix @ old_pagerank) + momentum
            if norm(pagerank - old_pagerank, ord=1) < convergence_threshold:
                break

        return pagerank.flatten()

# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import math
import numpy as np
import scipy as sp


class GNTK(object):
    """
    implement the Graph Neural Tangent Kernel
    """

    def __init__(self, num_layers, num_mlp_layers, jk, scale):
        """Configure graph aggregation and the ReLU covariance recurrence.

        :param num_layers: Number of graph-network layers including the input layer.
        :param num_mlp_layers: Number of MLP covariance updates per graph layer.
        :param jk: Whether to sum kernel contributions across graph layers
            using jumping knowledge instead of taking only the final layer.
        :param scale: Neighbor aggregation scaling, ``'uniform'`` or ``'degree'``.
        :raises AssertionError: If ``scale`` is not a supported scaling rule.
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        # self.num_agg_layers = num_agg_layers
        self.jk = jk
        self.scale = scale
        assert scale in ["uniform", "degree"]

    def __next_diag(self, S):
        """Update within-graph ReLU covariance and extract node standard deviations.

        :param S: Covariance or kernel matrix being propagated through the recurrence.
        :returns: Tuple ``(covariance, derivative_factor, standard_deviations)``.
        """
        diag = np.sqrt(np.diag(S))
        # tmp = diag[:, None]
        S = S / diag[:, None] / diag[None, :]
        S = np.clip(S, -1, 1)
        # dot sigma
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """Aggregate within-graph covariance through a sparse Kronecker adjacency operator.

        :param S: Covariance or kernel matrix being propagated through the recurrence.
        :param adj_block: Sparse Kronecker graph-pair adjacency operator.
        :param N: Node count of the corresponding covariance dimension.
        :param scale_mat: Scalar or node-pair normalization factor applied after aggregation.
        :returns: Square aggregated covariance matrix.
        """
        return adj_block.dot(S.reshape(-1)).reshape(N, N) * scale_mat

    def __next(self, S, diag1, diag2):
        """Update cross-graph ReLU covariance using saved variance scales.

        :param S: Covariance or kernel matrix being propagated through the recurrence.
        :param diag1: Standard-deviation vectors for the left and right node sets.
        :param diag2: Standard-deviation vectors for the left and right node sets.
        :returns: Tuple ``(updated_covariance, derivative_factor)``.
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = np.clip(S, -1, 1)
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS

    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """Aggregate cross-graph covariance through the graph-pair adjacency operator.

        :param S: Covariance or kernel matrix being propagated through the recurrence.
        :param adj_block: Sparse Kronecker graph-pair adjacency operator.
        :param N1: Node count of the corresponding covariance dimension.
        :param N2: Node count of the corresponding covariance dimension.
        :param scale_mat: Scalar or node-pair normalization factor applied after aggregation.
        :returns: Matrix of shape ``(N1, N2)`` after pairwise scaling.
        """
        return adj_block.dot(S.reshape(-1)).reshape(N1, N2) * scale_mat

    def show_memory(self, unit="KB", threshold=1):
        """Report variable memory usage.

        :param unit: Display unit: ``B``, ``KB``, ``MB`` or ``GB``.
        :param threshold: Only show variables whose memory usage is at least this value.
        """
        from sys import getsizeof

        scale = {"B": 1, "KB": 1024, "MB": 1048576, "GB": 1073741824}[unit]
        for i in list(globals().keys()):
            memory = eval("getsizeof({})".format(i)) // scale
            if memory >= threshold:
                print(i, memory)

    def diag(self, feat, A):
        """Collect node variance scales used by cross-graph GNTK recursion.

        :param feat: Node feature matrix with one row per graph node.
        :param A: SciPy adjacency matrix of the graph.
        :returns: List of node standard-deviation arrays in dense-layer recursion order.
        """
        N = A.shape[0]
        # tmp0 = np.sum(A, axis=1)
        # tmp1 = np.array(np.sum(A, axis=1) * np.sum(A, axis=0))
        if self.scale == "uniform":
            scale_mat = 1.0
        else:
            scale_mat = 1.0 / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))
            # assert False

        diag_list = []
        adj_block = sp.sparse.kron(A, A)
        sigma = np.matmul(feat, feat.T)
        # print(sigma)
        sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
        # print(sigma)
        ntk = np.copy(sigma)

        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_layers - 1:
                # for agg_layer in range(self.num_agg_layers):
                #    sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self.__adj_diag(ntk, adj_block, N, scale_mat)
        return diag_list

    def gntk(self, feat1, feat2, diag_list1, diag_list2, A1, A2):
        """Compute the graph-pair GNTK and retain its intermediate recurrences.

        :param feat1: First graph's node features with shape ``(num_nodes1, d)``.
        :param feat2: Second graph's node features with the same feature width.
        :param diag_list1: First graph's node standard deviations returned by ``diag``.
        :param diag_list2: Second graph's node standard deviations returned by ``diag``.
        :param A1: First graph's SciPy adjacency matrix.
        :param A2: Second graph's SciPy adjacency matrix.
        :returns: Tuple ``(kernel_value, covariances, tangent_kernels, derivatives)``.
            The scalar kernel value uses the configured jumping-knowledge rule;
            the remaining values are lists of intermediate node-pair matrices.
        """

        n1 = A1.shape[0]
        n2 = A2.shape[0]

        if self.scale == "uniform":
            scale_mat = 1.0
        else:
            scale_mat = 1.0 / np.array(np.sum(A1, axis=1) * np.sum(A2, axis=0))

        adj_block = sp.sparse.kron(A1, A2)

        jump_ntk = 0
        sigma = np.matmul(feat1, feat2.T)
        jump_ntk += sigma
        sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
        ntk = np.copy(sigma)

        list_sigma = []
        list_ntk = []
        list_dotsigma = []
        list_sigma.append(ntk)

        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(
                    sigma,
                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer],
                )
                ntk = ntk * dot_sigma + sigma

            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self.__adj(ntk, adj_block, n1, n2, scale_mat)

            list_sigma.append(sigma)
            list_ntk.append(ntk)
            list_dotsigma.append(dot_sigma)

        if self.jk:
            return np.sum(jump_ntk) * 2, list_sigma, list_ntk, list_dotsigma
        else:
            return np.sum(ntk) * 2, list_sigma, list_ntk, list_dotsigma

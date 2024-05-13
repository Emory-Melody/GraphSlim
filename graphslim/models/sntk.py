import math

import torch
import torch.nn as nn


class StructureBasedNeuralTangentKernel(nn.Module):
    def __init__(self, K=2, L=2, scale='add'):
        super(StructureBasedNeuralTangentKernel, self).__init__()
        self.K = K
        self.L = L
        self.scale = scale

    def sparse_kron(self, A, B):
        """
        A, B: torch.sparse.FloatTensor of shape (m, n) and (p, q)
        Returns: the Kronecker product of A and B
        """
        m, n = A.shape
        p, q = B.shape
        n_A = A._nnz()
        n_B = B._nnz()

        indices_A = A.coalesce().indices()
        indices_B = B.coalesce().indices()
        indices_A[0, :] = indices_A[0, :] * p
        indices_A[1, :] = indices_A[1, :] * q

        indices = (indices_A.repeat(n_B, 1) + indices_B.t().reshape(2 * n_B, 1))
        ind_row = indices.index_select(0, torch.arange(start=0, end=2 * n_B, step=2, device=A.device)).reshape(-1)
        ind_col = indices.index_select(0, torch.arange(start=1, end=2 * n_B, step=2, device=A.device)).reshape(-1)

        new_ind = torch.cat((ind_row, ind_col)).reshape(2, n_A * n_B)
        values = torch.ones(n_A * n_B).to(A.device)
        new_shape = (m * p, n * q)

        return torch.sparse_coo_tensor(new_ind, values, new_shape)

    def aggr(self, S, aggr_optor, n1, n2, scale_mat):
        S = torch.sparse.mm(aggr_optor, S.reshape(-1)[:, None]).reshape(n1, n2) * scale_mat
        # S += 1e-9
        return S

    def update_sigma(self, S, diag1, diag2):
        S = S / diag1[:, None] / diag2[None, :]
        S = torch.clip(S, -0.9999, 0.9999)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        degree_sigma = (math.pi - torch.arccos(S)) / math.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, degree_sigma

    def update_diag(self, S):
        diag = torch.sqrt(torch.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = torch.clip(S, -0.9999, 0.9999)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        S = S * diag[:, None] * diag[None, :]
        return S, diag

    def diag(self, g, E):
        n = E.shape[0]
        aggr_optor = self.sparse_kron(E, E)
        if self.scale == 'add':
            scale_mat = 1.
        else:
            scale_mat = (1. / torch.sparse.sum(aggr_optor, 1).to_dense()).reshape(n, n)
        diag_list = []
        sigma = torch.matmul(g, g.t())
        for k in range(self.K):
            sigma = self.aggr(sigma, aggr_optor, n, n, scale_mat)
            sigma, diag = self.update_diag(sigma)
            diag_list.append(diag)
        return diag_list

    def nodes_gram(self, g1, g2, E1, E2):
        n1, n2 = len(g1), len(g2)
        aggr_optor = self.sparse_kron(E1, E2)

        if self.scale == 'add':
            scale_mat = 1.
        else:
            scale_mat = (1. / torch.sparse.sum(aggr_optor, 1).to_dense()).reshape(n1, n2)

        sigma = torch.matmul(g1, g2.t())
        theta = sigma
        diag_list1, diag_list2 = self.diag(g1, E1), self.diag(g2, E2)

        for k in range(self.K):
            sigma = self.aggr(sigma, aggr_optor, n1, n2, scale_mat)
            theta = self.aggr(theta, aggr_optor, n1, n2, scale_mat)

            for l in range(self.L):
                sigma, degree_sigma = self.update_sigma(sigma, diag_list1[k], diag_list2[k])
                theta = theta * degree_sigma + sigma

        return theta

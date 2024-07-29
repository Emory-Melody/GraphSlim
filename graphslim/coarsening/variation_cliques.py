import copy

import numpy as np
import scipy as sp
import torch
from pygsp import graphs
from torch_geometric.utils import to_dense_adj

from graphslim.coarsening.utils import contract_variation_edges, contract_variation_linear, get_proximity_measure, \
    matching_optimal, matching_greedy, get_coarsening_matrix, coarsen_matrix, coarsen_vector, zero_diag
from graphslim.dataset.convertor import pyg2gsp, csr2ei, ei2csr
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *
from graphslim.utils import one_hot, to_tensor
from graphslim.coarsening.coarsening_base import Coarsen


class VariationCliques(Coarsen):
    def __init__(self, setting, data, args, **kwargs):
        super(Coarsen, self).__init__(setting, data, args, **kwargs)
    def coarsen(self, G):
        K = 10
        r = self.args.reduction_rate
        max_levels = 10
        Uk = None
        lk = None
        max_level_r = 0.99,
        r = np.clip(r, 0, 0.999)
        G0 = G
        N = G.N

        # current and target graph sizes
        n, n_target = N, np.ceil((1 - r) * N)

        C = sp.sparse.eye(N, format="csc")
        Gc = G

        Call, Gall = [], []
        Gall.append(G)
        method = "variation_cliques"
        for level in range(1, max_levels + 1):

            G = Gc

            # how much more we need to reduce the current graph
            r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

            if "variation" in method:

                if level == 1:
                    if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                        mask = lk < 1e-10
                        lk[mask] = 1
                        lsinv = lk ** (-0.5)
                        lsinv[mask] = 0
                        B = Uk[:, :K] @ np.diag(lsinv[:K])
                    else:
                        offset = 2 * max(G.dw)
                        T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                        lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                        lk = (offset - lk)[::-1]
                        Uk = Uk[:, ::-1]
                        mask = lk < 1e-10
                        lk[mask] = 1
                        lsinv = lk ** (-0.5)
                        lsinv[mask] = 0
                        B = Uk @ np.diag(lsinv)
                    A = B
                else:
                    B = iC.dot(B)
                    d, V = np.linalg.eig(B.T @ (G.L).dot(B))
                    mask = d == 0
                    d[mask] = 1
                    dinvsqrt = (d + 1e-9) ** (-1 / 2)
                    dinvsqrt[mask] = 0
                    A = B @ np.diag(dinvsqrt) @ V

                coarsening_list = contract_variation_linear(
                    G, K=K, A=A, r=r_cur, mode=method
                )

            iC = get_coarsening_matrix(G, coarsening_list)

            if iC.shape[1] - iC.shape[0] <= 2:
                break  # avoid too many levels for so few nodes

            C = iC.dot(C)
            Call.append(iC)

            Wc = zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
            Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

            if not hasattr(G, "coords"):
                Gc = graphs.Graph(Wc)
            else:
                Gc = graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
            Gall.append(Gc)

            n = Gc.N

            if n <= n_target:
                break

        return C, Gc, Call, Gall

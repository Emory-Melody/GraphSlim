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


class AffinityGs(Coarsen):

    def coarsen(self, G):
        """
        This function provides a common interface for coarsening algorithms that contract subgraphs

        Parameters
        ----------
        G : pygsp Graph
        K : int
            The size of the subspace we are interested in preserving.
        r : float between (0,1)
            The desired reduction defined as 1 - n/N.
        method : String
            ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']

        Returns
        -------
        C : np.array of size n x N
            The coarsening matrix.
        Gc : pygsp Graph
            The smaller graph.
        Call : list of np.arrays
            Coarsening matrices for each level
        Gall : list of (n_levels+1) pygsp Graphs
            All graphs involved in the multilevel coarsening

        Example
        -------
        C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
        """
        K = 10
        r = 0.5
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
        method = "affinity_GS"
        algorithm=self.args.coarsen_strategy
        # algorithm = "greedy"
        for level in range(1, max_levels + 1):

            G = Gc

            # how much more we need to reduce the current graph
            r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

            weights = get_proximity_measure(G, method, K=K)

            if algorithm == "optimal":
                # the edge-weight should be light at proximal edges
                weights = -weights
                if "rss" not in method:
                    weights -= min(weights)
                coarsening_list = matching_optimal(G, weights=weights, r=r_cur)

            elif algorithm == "greedy":
                coarsening_list = matching_greedy(G, weights=weights, r=r_cur)
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

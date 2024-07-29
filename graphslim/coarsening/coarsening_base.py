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


class Coarsen:
    """
    Graph Coarsening

    This class provides methods to coarsen a graph, which involves reducing the number of nodes while
    preserving certain properties of the original graph.

    Parameters
    ----------
    setting : str
        The setting for coarsening.
    data : object
        The data to be coarsened.
    args : Namespace
        Arguments containing configurations and device information.
    **kwargs : dict, optional
        Additional keyword arguments.

    Methods
    -------
    reduce(data, verbose=True, save=True)
        Reduces the data by coarsening the graph.
    coarsening(H)
        Performs the coarsening process on the graph H.
    process_coarsened(data, candidate, C_list, Gc_list)
        Processes the coarsened graphs and returns the features, labels, masks, and edges.
    """

    def __init__(self, setting, data, args, **kwargs):
        """
        Initializes the Coarsen object.

        Parameters
        ----------
        setting : str
            The setting for coarsening.
        data : object
            The data to be coarsened.
        args : Namespace
            Arguments containing configurations and device information.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.setting = setting
        self.args = args
        self.device = args.device
        # pass data for initialization

    @verbose_time_memory
    def reduce(self, data, verbose=True, save=True):
        """
        Reduces the data by coarsening the graph.

        Parameters
        ----------
        data : object
            The data to be reduced.
        verbose : bool, optional
            If True, prints verbose output. Defaults to True.
        save : bool, optional
            If True, saves the reduced data. Defaults to True.

        Returns
        -------
        object
            The reduced data with updated attributes `adj_syn`, `feat_syn`, and `labels_syn`.
        """
        args = self.args
        # setting = self.setting
        # device = self.device

        cpu_data = copy.deepcopy(data)

        if args.setting == 'trans':
            gps_graph = graphs.Graph(W=data.adj_full)
            candidate, C_list, Gc_list = self.coarsening(gps_graph)
            coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_edge = self.process_coarsened(
                cpu_data, candidate, C_list, Gc_list)

            train_idx = np.nonzero(coarsen_train_mask.numpy())[0]
            coarsen_features = coarsen_features[train_idx]
            coarsen_edge = ei2csr(coarsen_edge, coarsen_train_mask.shape[0])[np.ix_(train_idx, train_idx)]
            coarsen_train_labels = coarsen_train_labels[train_idx]
        else:
            gps_graph = graphs.Graph(W=data.adj_full)
            candidate, C_list, Gc_list = self.coarsening(gps_graph)
            coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_edge = self.process_coarsened(
                cpu_data, candidate, C_list, Gc_list)

            coarsen_edge = ei2csr(coarsen_edge, coarsen_train_mask.shape[0])

        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(coarsen_edge), coarsen_features, coarsen_train_labels
        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        return data

    def coarsening(self, H):
        """
        Performs the coarsening process on the given graph.

        Parameters
        ----------
        H : object
            The graph to be coarsened.

        Returns
        -------
        candidate : list
            List of subgraphs sorted by size.
        C_list : list
            List of coarsening matrices.
        Gc_list : list
            List of coarsened graphs.
        """
        if H.A.shape[-1] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(H.A.shape[-1], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1] if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        print('the number of subgraphs is', len(graphs))
        candidate = sorted(graphs, key=lambda x: len(x.info['orig_idx']), reverse=True)
        number = 0
        C_list = []
        Gc_list = []
        while number < len(candidate):
            H = candidate[number]
            if len(H.info['orig_idx']) > 10:
                C, Gc, Call, Gall = self.coarsen(H)
                C_list.append(C)
                Gc_list.append(Gc)
            number += 1
        return candidate, C_list, Gc_list

    def process_coarsened(self, data, candidate, C_list, Gc_list):
        """
        Processes the coarsened graph to extract relevant features and labels.

        Parameters
        ----------
        data : object
            The original data.
        candidate : list
            List of candidate subgraphs.
        C_list : list
            List of coarsening matrices.
        Gc_list : list
            List of coarsened graphs.

        Returns
        -------
        coarsen_features : torch.Tensor
            The coarsened features.
        coarsen_train_labels : torch.Tensor
            The coarsened training labels.
        coarsen_train_mask : torch.Tensor
            The coarsened training mask.
        coarsen_edge : torch.Tensor
            The coarsened edges.
        """
        train_mask = data.train_mask
        val_mask = data.val_mask

        n_classes = max(data.y) + 1
        if self.args.setting == 'trans':
            features = data.x
            labels = data.y
        else:
            features = data.x[train_mask]
            labels = data.y[train_mask]
        coarsen_node = 0
        number = 0
        coarsen_row = None
        coarsen_col = None
        coarsen_features = torch.Tensor([])
        coarsen_train_labels = torch.Tensor([])
        coarsen_train_mask = torch.Tensor([]).bool()

        while number < len(candidate):
            H = candidate[number]
            keep = H.info['orig_idx']
            H_features = features[keep]
            H_labels = labels[keep]
            if self.args.setting == "trans":
                H_train_mask = train_mask[keep]
            else:
                H_train_mask = torch.ones(size=(len(H_labels),))

            if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask) > 0:
                train_labels = one_hot(H_labels, n_classes)  # Shape: (H_labels.shape[0], n_classes)
                if self.args.setting == "trans":
                    train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
                C = C_list[number]
                Gc = Gc_list[number]

                new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
                mix_label = torch.FloatTensor(C.dot(train_labels))
                mix_label[mix_label > 0] = 1
                mix_mask = torch.sum(mix_label, dim=1)
                new_train_mask[mix_mask > 1] = False

                coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
                coarsen_train_labels = torch.cat(
                    [coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
                coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)

                if coarsen_row is None:
                    coarsen_row = Gc.W.tocoo().row
                    coarsen_col = Gc.W.tocoo().col
                else:
                    current_row = Gc.W.tocoo().row + coarsen_node
                    current_col = Gc.W.tocoo().col + coarsen_node
                    coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                    coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
                coarsen_node += Gc.W.shape[0]
                print(coarsen_node)

            number += 1

        coarsen_edge = torch.from_numpy(np.array([coarsen_row, coarsen_col])).long()
        coarsen_train_labels = coarsen_train_labels.long()

        return coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_edge

    def coarsen(self, G):
        """
        This function provides a common interface for coarsening algorithms that contract subgraphs.

        Parameters
        ----------
        G : pygsp Graph
            The graph to be coarsened.

        Returns
        -------
        C : np.array of size n x N
            The coarsening matrix.
        Gc : pygsp Graph
            The smaller, coarsened graph.
        Call : list of np.arrays
            Coarsening matrices for each level.
        Gall : list of pygsp Graphs
            All graphs involved in the multilevel coarsening.

        Example
        -------
        >>> C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
        """

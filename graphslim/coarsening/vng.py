import copy

import numpy as np
import torch
from sklearn.cluster import KMeans

from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import GCN
from graphslim.utils import one_hot


class VNG:
    """
    A class that implements Virtual Node Graph (VNG) reduction for coarsening graphs.
    Refer to paper "Serving Graph Compression for Graph Neural Networks" https://openreview.net/forum?id=T-qVtA3pAxG.

    Parameters
    ----------
    setting : str
        Configuration setting.
    data : object
        Data object containing the graph and feature information.
    args : object
        Arguments containing various settings for the coarsening process.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, setting, data, args, **kwargs):
        self.setting = setting
        self.args = args
        self.device = args.device
        # Pass data for initialization

    @verbose_time_memory
    def reduce(self, data, verbose=False):
        """
        Reduces the data by applying Virtual Node Graph (VNG) method.

        Parameters
        ----------
        data : object
            The data to be reduced.
        verbose : bool, optional
            If True, prints verbose output. Defaults to False.

        Returns
        -------
        object
            The reduced data with synthesized adjacency, features, and labels.
        """
        args = self.args
        setting = self.setting
        cpu_data = copy.deepcopy(data)

        if args.setting == 'trans':
            model = eval(args.eval_model)(data.feat_full.shape[1], args.hidden, data.nclass, args).to(self.device)
            model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose, setting=args.setting,
                               reduced=False)
            embeds = model.predict(data.feat_full, data.adj_full, output_layer_features=True)
            embeds = [embed[data.train_mask] for embed in embeds]
            labels = one_hot(data.labels_train, data.nclass)

            coarsen_edge, coarsen_features, coarsen_labels = self.vng(embeds, data.adj_train, labels)

        else:
            model = eval(args.eval_model)(data.feat_full.shape[1], args.hidden, data.nclass, args).to(self.device)
            model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose, setting=args.setting,
                               reduced=False, reindex=True)
            model.eval()
            embeds = model.predict(data.feat_train, data.adj_train, output_layer_features=True)
            embeds = [embed for embed in embeds]
            labels = one_hot(data.labels_train, data.nclass)

            coarsen_edge, coarsen_features, coarsen_labels = self.vng(embeds, data.adj_train, labels)

        data.adj_syn, data.feat_syn, data.labels_syn = coarsen_edge, coarsen_features, coarsen_labels

        save_reduced(coarsen_edge, coarsen_features, coarsen_labels, args)

        return data

    def vng(self, embeds, adj, labels, verbose=False):
        """
        Virtual Node Graph (VNG) method to coarsen the graph.

        Parameters
        ----------
        embeds : list of tensors
            List of embeddings.
        adj : tensor
            Adjacency matrix.
        labels : tensor
            One-hot encoded labels.
        verbose : bool, optional
            If True, prints verbose output. Defaults to False.

        Returns
        -------
        tuple
            A tuple containing:

            - coarsen_edge : tensor
                Coarsened adjacency matrix.
            - coarsen_features : tensor
                Coarsened features.
            - coarsen_labels : tensor
                Coarsened labels.
        """
        X_tr_head = torch.concat(embeds, dim=1).cpu().detach().numpy()
        X_tr_0 = embeds[0].cpu().detach().numpy()
        A_tr = adj

        column_sum = np.sum(A_tr, axis=0)
        column_sum = np.asarray(column_sum).reshape(-1)
        for i in range(X_tr_0.shape[0]):  # Prevent column_sum from being zero
            if column_sum[i] == 0:
                column_sum[i] = 1

        num_coarsen_node = int(self.args.reduction_rate * X_tr_0.shape[0])

        # PROPAGATION FROM VIRTUAL REPRESENTATIVE NODES TO TESTING NODES
        kmeans = KMeans(n_clusters=num_coarsen_node, random_state=2024, n_init=1)
        kmeans.fit(X_tr_head, sample_weight=column_sum)

        E = np.zeros((num_coarsen_node, X_tr_0.shape[0]))
        M = np.zeros((num_coarsen_node, X_tr_0.shape[0]))
        for i in range(X_tr_0.shape[0]):
            cluster_label = kmeans.labels_[i]
            E[cluster_label, i] = column_sum[i]
            M[cluster_label, i] = 1
        row_sums = E.sum(axis=1)
        E = E / row_sums[:, np.newaxis]

        X_vr_0 = E @ X_tr_0

        if verbose:
            print("E: ", E.shape)
            print("X_vr_0: ", X_vr_0.shape)

        # PROPAGATION BETWEEN VIRTUAL REPRESENTATIVE NODES
        P = E @ X_tr_head
        Q = E @ A_tr @ X_tr_head
        Up, Sp, Vtp = np.linalg.svd(P, full_matrices=False)
        A_vr = Q @ Vtp.T @ np.diag(1 / Sp) @ Up.T

        if verbose:
            print("P: ", P.shape)
            print("Q: ", Q.shape)
            print("Up: ", Up.shape)
            print("Sp: ", Sp.shape)
            print("Vtp: ", Vtp.shape)
            print("Vp: ", Vtp.T.shape)
            print("A_vr Sparsity", len(np.nonzero(A_vr)[0]) / A_vr.shape[0] ** 2)
            print("A_vr: ", A_vr.shape)

        coarsen_features = torch.from_numpy(X_vr_0).float()
        coarsen_edge = torch.from_numpy(A_vr).float()
        coarsen_labels = torch.argmax(torch.FloatTensor(M.dot(labels)), dim=1).long()

        return coarsen_edge, coarsen_features, coarsen_labels


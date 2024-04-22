import copy
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from graphslim.dataset.utils import save_reduced
from graphslim.models import GCN
from graphslim.utils import one_hot, cal_storage


class VNG:
    def __init__(self, setting, data, args, **kwargs):
        self.setting = setting
        self.args = args
        self.device = args.device
        # pass data for initialization

    def reduce(self, data, verbose=True):
        if verbose:
            start = time.perf_counter()

        args = self.args
        setting = self.setting
        # device = self.device

        cpu_data = copy.deepcopy(data)

        if args.setting == 'trans':
            model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                        weight_decay=args.weight_decay).to(args.device)
            model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose, setting='trans')
            embeds = model.predict(data.feat_full, data.adj_full, output_layer_features=True)
            embeds = [embed[data.train_mask] for embed in embeds]
            labels = one_hot(data.labels_train, data.nclass)

            coarsen_edge, coarsen_features, coarsen_labels = self.vng(embeds, data.adj_train, labels)

        else:
            model = GCN(nfeat=data.feat_full.shape[1], nhid=args.hidden, nclass=data.nclass, device=args.device,
                        weight_decay=args.weight_decay).to(args.device)
            model.fit_with_val(data, train_iters=args.eval_epochs, verbose=verbose, setting='ind', reindex=True)
            model.eval()
            embeds = model.predict(data.feat_train, data.adj_train, output_layer_features=True)
            embeds = [embed for embed in embeds]
            labels = one_hot(data.labels_train, data.nclass)

            coarsen_edge, coarsen_features, coarsen_labels = self.vng(embeds, data.adj_train, labels)

        data.adj_syn, data.feat_syn, data.labels_syn = coarsen_edge, coarsen_features, coarsen_labels

        if verbose:
            end = time.perf_counter()
            print("Reduce Time: ", (end - start) * 1000, "ms")
            cal_storage(data, setting=args.setting)

        save_reduced(coarsen_edge, coarsen_features, coarsen_labels, args)

        return data

    def vng(self, embeds, adj, labels, verbose=False):
        X_tr_head = torch.concat(embeds, dim=1).cpu().detach().numpy()

        X_tr_0 = embeds[0].cpu().detach().numpy()

        A_tr = adj

        column_sum = np.sum(A_tr, axis=0)
        column_sum = np.asarray(column_sum).reshape(-1)
        for i in range(X_tr_0.shape[0]):  # 防止出现column_sum等于0的情况
            if column_sum[i] == 0:
                column_sum[i] = 1

        num_coarsen_node = int(self.args.reduction_rate * X_tr_0.shape[0])

        '''
        PROPAGATION FROM VIRTUAL REPRESENTATIVE NODES TO TESTING NODES
        '''

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

        '''
        PROPAGATION BETWEEN VIRTUAL REPRESENTATIVE NODES
        '''

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

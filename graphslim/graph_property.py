import os
import sys
import scipy

from sklearn.metrics import pairwise_distances

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from graphslim.configs import *
from graphslim.dataset import *
from graphslim.evaluation.utils import sparsify
import matplotlib.pyplot as plt
import logging
import networkx as nx
import numpy as np
import torch
from graphslim.utils import normalize_adj
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def calculate_homophily(y, adj):
    # Convert dense numpy array to sparse matrix if necessary
    if isinstance(adj, np.ndarray):
        adj = scipy.sparse.csr_matrix(adj)

    if not scipy.sparse.isspmatrix_csr(adj):
        adj = adj.tocsr()

    # Binarize the adjacency matrix (assuming adj contains weights)
    adj.data = (adj.data > 0.5).astype(int)

    # Ensure y is a 1D array
    y = np.squeeze(y)

    # Get the indices of the non-zero entries in the adjacency matrix
    edge_indices = adj.nonzero()

    # Get the labels of the source and target nodes for each edge
    src_labels = y[edge_indices[0]]
    tgt_labels = y[edge_indices[1]]

    # Calculate the homophily as the fraction of edges connecting nodes of the same label
    same_label = src_labels == tgt_labels
    homophily = np.mean(same_label)

    return homophily


# def davies_bouldin_index(X, labels):
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     unique_labels = np.unique(labels)
#     n_clusters = len(unique_labels)
#     cluster_kmeans = [X[labels == k] for k in unique_labels]
#
#     centroids = []
#     scatters = []
#
#     for cluster in cluster_kmeans:
#         if cluster.shape[0] > 0:
#             centroid = np.mean(cluster, axis=0)
#             centroids.append(centroid)
#             scatter = np.mean(pairwise_distances(cluster, centroid.reshape(1, -1)))
#             scatters.append(scatter)
#         else:
#             centroids.append(None)
#             scatters.append(None)
#
#     if len([c for c in centroids if c is not None]) < 2:
#         raise ValueError("Not enough non-empty clusters to calculate Davies-Bouldin index.")
#
#     db_index = 0
#     for i in range(n_clusters):
#         if centroids[i] is None:
#             continue
#         max_ratio = 0
#         for j in range(n_clusters):
#             if i != j and centroids[j] is not None:
#                 d_ij = np.linalg.norm(centroids[i] - centroids[j])
#                 ratio = (scatters[i] + scatters[j]) / d_ij
#                 max_ratio = max(max_ratio, ratio)
#         db_index += max_ratio
#
#     db_index /= n_clusters
#     return db_index


def graph_property(adj, feat, label):
    spe_list = []
    clu_list = []
    den_list = []
    hom_list = []
    db_list = []
    db_agg_list = []
    if len(adj.shape) == 3:
        for i in range(adj.shape[0]):
            G = nx.from_numpy_array(adj[i])
            laplacian_matrix = nx.laplacian_matrix(G).astype(np.float32)
            k = 1
            eigenvalues, _ = scipy.sparse.linalg.eigsh(laplacian_matrix, k=k, which='LM')

            # The largest eigenvalue is the spectral radius
            spectral_radius = max(eigenvalues)
            spe_list.append(spectral_radius)
            # spectral_min = min(eigenvalues[eigenvalues > 0])

            cluster_coefficient = nx.average_clustering(G)
            clu_list.append(cluster_coefficient)

            density = nx.density(G)
            den_list.append(density)
            # sparsity = 1 - density

            homophily = calculate_homophily(label, adj)
            hom_list.append(homophily)
            db_index = davies_bouldin_score(feat, label)
            db_list.append(db_index)
            adj = normalize_adj(adj)

            db_index_agg = davies_bouldin_score(adj @ adj @ feat, label)
            db_agg_list.append(db_index_agg)
        spe_list = np.array(spe_list)
        clu_list = np.array(clu_list)
        den_list = np.array(den_list)
        hom_list = np.array(hom_list)
        db_list = np.array(db_list)
        db_agg_list = np.array(db_agg_list)
        args.logger.info(f"Average Density %: {np.mean(den_list) * 100}")
        args.logger.info(f"Average Spectral Radius: {np.mean(spe_list)}")
        args.logger.info(f"Average Cluster Coefficient: {np.mean(clu_list)}")
        args.logger.info(f"Average Homophily: {np.mean(hom_list)}")
        args.logger.info(f"Average Davies-Bouldin Index: {np.mean(db_list)}")
        args.logger.info(f"Average Davies-Bouldin Index AGG: {np.mean(db_agg_list)}")


    else:
        G = nx.from_numpy_array(adj)

        laplacian_matrix = nx.laplacian_matrix(G).astype(np.float32)

        # Compute the largest eigenvalue using sparse linear algebra
        k = 1  # number of eigenvalues and eigenvectors to compute
        eigenvalues, _ = scipy.sparse.linalg.eigsh(laplacian_matrix, k=k, which='LM')

        # The largest eigenvalue is the spectral radius
        spectral_radius = max(eigenvalues)
        # spectral_min = min(eigenvalues[eigenvalues > 0])

        cluster_coefficient = nx.average_clustering(G)

        density = nx.density(G)
        # sparsity = 1 - density

        homophily = calculate_homophily(label, adj)
        db_index = davies_bouldin_score(feat, label)
        adj = normalize_adj(adj)

        db_index_agg = davies_bouldin_score(adj @ adj @ feat, label)
        # print("Degree Distribution:", degree_distribution)
        args.logger.info(f"Density %: {density * 100}")
        args.logger.info(f"Spectral Radius: {spectral_radius}")
        # print("Spectral Min:", spectral_min)
        args.logger.info(f"Cluster Coefficient: {cluster_coefficient}")
        # print("Density:", density)
        args.logger.info(f"Homophily: {homophily}")
        args.logger.info(f"Davies-Bouldin Index: {db_index}")
        args.logger.info(f"Davies-Bouldin Index AGG: {db_index_agg}")


if __name__ == '__main__':
    args = cli(standalone_mode=False)
    args.device = 'cpu'

    if args.origin:
        graph = get_dataset(args.dataset, args)
        if args.setting == 'ind':
            adj, feat, label = graph.adj_train, graph.feat_train, graph.labels_train
        else:
            adj, feat, label = graph.adj_full, graph.feat_full, graph.labels_full
        feat = feat.numpy()
        label = label.numpy()
        graph_property(adj, feat, label)
    method_list = ['gcond', 'doscond', 'msgc', 'sgdd', 'geom', 'gcsntk']
    for args.method in method_list:
        adj_syn, feat, label = load_reduced(args)
        label = label[:adj_syn.shape[0]]
        if args.method in ['geom', 'gcsntk']:
            label = torch.argmax(label, dim=1)
        adj_syn = sparsify('GCN', adj_syn, args, verbose=args.verbose)
        adj = adj_syn.numpy()
        label = label.numpy()
        feat = feat.numpy()
        print(f'========{args.method}========')
        graph_property(adj, feat, label)

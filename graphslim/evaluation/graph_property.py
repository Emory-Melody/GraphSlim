import os
import sys

if os.path.abspath('') not in sys.path:
    sys.path.append(os.path.abspath(''))
from graphslim.configs import *
from graphslim.dataset import *
from graphslim.evaluation.utils import sparsify
import logging
import networkx as nx
import numpy as np
import torch


def calculate_homophily(y, adj):
    y = y.squeeze()
    edge_indices = np.asarray(adj.nonzero())
    src_labels = y[edge_indices[0]]
    tgt_labels = y[edge_indices[1]]
    same_label = src_labels == tgt_labels
    homophily = same_label.mean()

    return homophily


if __name__ == '__main__':
    args = cli(standalone_mode=False)

    args.device = 'cpu'
    if args.origin:
        graph = get_dataset(args.dataset, args)
        if args.setting == 'ind':
            adj, label = graph.adj_train, graph.labels_train
        else:
            adj, label = graph.adj_full, graph.labels_full
        adj = adj.toarray()
        label = label.numpy()
    else:
        save_path = f'checkpoints/reduced_graph/{args.method}'
        adj_syn = torch.load(
            f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        label = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        adj_syn = sparsify('GCN', adj_syn, args, verbose=args.verbose)
        adj = (adj_syn > 0.5).int().numpy()
        label = label.numpy()

    G = nx.from_numpy_array(adj)

    degree_distribution = nx.degree_histogram(G)

    laplacian_matrix = nx.laplacian_matrix(G)
    eigenvalues = np.linalg.eigvals(laplacian_matrix.A)
    spectral_radius = max(eigenvalues)
    spectral_min = min(eigenvalues[eigenvalues > 0])

    cluster_coefficient = nx.average_clustering(G)

    density = nx.density(G)
    sparsity = 1 - density

    # 同质性
    homophily = calculate_homophily(label, adj)

    print("Degree Distribution:", degree_distribution)
    print("Spectral Radius:", spectral_radius)
    print("Spectral Min:", spectral_min)
    print("Cluster Coefficient:", cluster_coefficient)
    # print("Density:", density)
    print("Sparsity:", sparsity)
    print("Homophily:", homophily)

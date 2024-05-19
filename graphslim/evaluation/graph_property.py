import os
import sys

if os.path.abspath('') not in sys.path:
    sys.path.append(os.path.abspath(''))
from graphslim.configs import *
from graphslim.dataset import *
from graphslim.evaluation.utils import sparsify
import matplotlib.pyplot as plt
import logging
import networkx as nx
import numpy as np
import torch


def calculate_homophily(y, adj):
    adj = (adj > 0.5).astype(int)
    y = y.squeeze()
    edge_indices = np.asarray(adj.nonzero())
    src_labels = y[edge_indices[0]]
    tgt_labels = y[edge_indices[1]]
    same_label = src_labels == tgt_labels
    homophily = same_label.mean()

    return homophily


def plot_normalized_degree_distribution(degree_frequencies, graph_names):
    plt.figure(figsize=(6, 6))

    markers = ['o', 's', '^', 'D', 'v']  # Different marker styles
    colors = ['b', 'g', 'r', 'c', 'm']  # Different colors

    for i, (freq, name) in enumerate(zip(degree_frequencies, graph_names)):
        total_nodes = sum(freq)
        degrees = list(range(len(freq)))
        max_degree = max(degrees)
        normalized_degrees = [d / max_degree for d in degrees]
        normalized_freq = [f / total_nodes for f in freq]
        plt.scatter(normalized_degrees, normalized_freq, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], label=name, s=100, alpha=0.75, edgecolors='w')

    plt.xlabel('Degree')
    plt.ylabel('Normalized Frequency')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def graph_property(adj, label):
    G = nx.from_numpy_array(adj)

    degree_distribution = nx.degree_histogram(G)

    laplacian_matrix = nx.laplacian_matrix(G)
    eigenvalues = np.linalg.eigvals(laplacian_matrix.A)
    spectral_radius = max(eigenvalues)
    spectral_min = min(eigenvalues[eigenvalues > 0])

    cluster_coefficient = nx.average_clustering(G)

    density = nx.density(G)
    # sparsity = 1 - density

    # 同质性
    homophily = calculate_homophily(label, adj)

    # print("Degree Distribution:", degree_distribution)
    print("Spectral Radius:", spectral_radius)
    print("Spectral Min:", spectral_min)
    print("Cluster Coefficient:", cluster_coefficient)
    # print("Density:", density)
    print("Density %:", density * 100)
    print("Homophily:", homophily)
    return degree_distribution


if __name__ == '__main__':

    args = cli(standalone_mode=False)

    args.device = 'cpu'
    graph = get_dataset(args.dataset, args)
    if args.setting == 'ind':
        adj, label = graph.adj_train, graph.labels_train
    else:
        adj, label = graph.adj_full, graph.labels_full
    adj = adj.toarray()
    label = label.numpy()
    degree_distribution_origin = graph_property(adj, label)
    save_path = f'checkpoints/reduced_graph/{args.method}'
    adj_syn = torch.load(
        f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
    label = torch.load(
        f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
    adj_syn = sparsify('GCN', adj_syn, args, verbose=args.verbose)
    adj = adj_syn.numpy()
    label = label.numpy()
    degree_distribution = graph_property(adj, label)

    # plot_normalized_degree_distribution([degree_distribution, degree_distribution_origin],
    #                                     [args.dataset + '_reduced', args.dataset + '_origin'])

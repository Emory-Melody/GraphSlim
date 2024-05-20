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


def plot_normalized_degree_distribution(degree_frequencies, graph_name, method_list):
    plt.figure(figsize=(6, 6))
    graph_names = [graph_name + '_' + n for n in method_list]
    graph_names.append(graph_name)

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
    # plt.show()
    if not os.path.exists('evaluation/degree_distribution/'):
        os.makedirs('evaluation/degree_distribution/')
    plt.savefig(f'evaluation/degree_distribution/{graph_name}.pdf', format='pdf')


def graph_property(adj, feat, label):
    G = nx.from_numpy_array(adj)
    degree_distribution = nx.degree_histogram(G)

    return degree_distribution


if __name__ == '__main__':

    args = cli(standalone_mode=False)

    args.device = 'cpu'
    graph = get_dataset(args.dataset, args)
    if args.setting == 'ind':
        adj, feat, label = graph.adj_train, graph.feat_train, graph.labels_train
    else:
        adj, feat, label = graph.adj_full, graph.feat_full, graph.labels_full
    adj = adj.toarray()
    degree_distribution_origin = graph_property(adj, feat, label)
    dd_list = []
    method_list = ['gcond', 'doscond', 'msgc', 'sgdd']
    for args.method in method_list:

        save_path = f'checkpoints/reduced_graph/{args.method}'
        adj_syn = torch.load(
            f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        label = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        feat = torch.load(
            f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        if args.method == 'msgc':
            adj_syn = adj_syn[0]
            label = label[:adj_syn.shape[0]]
        adj_syn = sparsify('GCN', adj_syn, args, verbose=args.verbose)
        adj = adj_syn.numpy()
        degree_distribution = graph_property(adj, feat, label)
        dd_list.append(degree_distribution)
    dd_list.append(degree_distribution_origin)

    plot_normalized_degree_distribution(np.array(dd_list), args.dataset, method_list)

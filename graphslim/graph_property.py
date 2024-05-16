import os
import sys

if os.path.abspath('') not in sys.path:
    sys.path.append(os.path.abspath(''))
from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.condensation import *
from graphslim.dataset import *
import logging
import networkx as nx
import numpy as np
import torch

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)

save_path = f'checkpoints/reduced_graph/{args.method}'
adj_syn = torch.load(
    f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

print(graph.edge_index)
print(adj_syn)

adjacency_matrix = [[0, 1, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 0]]

# 转换为 NetworkX 图对象
G = nx.from_numpy_array(np.array(adjacency_matrix))

# 度分布
degree_distribution = nx.degree_histogram(G)

# 谱图
laplacian_matrix = nx.laplacian_matrix(G)
eigenvalues = np.linalg.eigvals(laplacian_matrix.A)
spectral_radius = max(eigenvalues)

# 聚类系数
cluster_coefficient = nx.average_clustering(G)

# 密度和稀疏度
density = nx.density(G)
sparsity = 1 - density

# 同质性
homophily = nx.attribute_assortativity_coefficient(G, 'attribute_name')

# 输出结果
print("Degree Distribution:", degree_distribution)
print("Spectral Radius:", spectral_radius)
print("Cluster Coefficient:", cluster_coefficient)
print("Density:", density)
print("Sparsity:", sparsity)
print("Homophily:", homophily)

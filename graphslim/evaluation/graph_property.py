import os
import sys
import scipy

from sklearn.metrics import pairwise_distances
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from graphslim.config import *
from graphslim.dataset import *
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import logging
import networkx as nx
import numpy as np
from graphslim.dataset import *
from graphslim.evaluation.utils import calculate_homophily
from graphslim.utils import normalize_adj
from sklearn.metrics import davies_bouldin_score
from scipy.sparse.csgraph import laplacian


class PropertyEvaluator:
    '''
    Class for evaluating graph properties of original and synthetic
    graphs, such as density, Laplacian space trace, spectral radius,
    cluster coefficient, homophily, and Davies-Bouldin index

    Parameters
    ----------
    args : object
        Arguments object containing the command-line arguments
    '''

    def __init__(self, args):

        self.args = args

    def evaluate(self, data, reduced=True, model_type='GCN'):
        '''
        Evaluates the graph properties of the original or synthetic graph

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data
        reduced : bool
            Whether to evaluate the reduced graph or the full graph
        model_type : str
            The type of model to use for the evaluation
        '''
        if reduced:
            self.feat, self.adj, self.label = get_syn_data(data, self.args, model_type=model_type)
            self.feat, self.adj, self.label = self.feat.cpu().numpy(), self.adj.cpu().numpy(), self.label.cpu().numpy()
        else:
            self.feat, self.adj, self.label = data.feat_full.numpy(), data.adj_full, data.labels_full.numpy()

        if self.args.with_structure:
            self.graph_property()
        else:
            self.graph_property_no_structure()

    def graph_property(self):

        args = self.args
        adj, feat, label = self.adj, self.feat, self.label
        eigtrace_list = []
        spe_list = []
        clu_list = []
        den_list = []
        hom_list = []
        db_list = []
        db_agg_list = []
        if len(adj.shape) == 3:
            for i in range(adj.shape[0]):
                ad = adj[i]
                G = nx.from_numpy_array(ad)
                # laplacian_matrix = nx.laplacian_matrix(G).astype(np.float32)
                laplacian_matrix = laplacian(ad, normed=True)
                eigenvalues, eigenvector = eigsh(laplacian_matrix)
                eig = feat.T @ eigenvector @ eigenvector.T @ feat
                trace = np.trace(eig) / feat.shape[1]
                eigtrace_list.append(trace)

                # The largest eigenvalue is the spectral radius
                spectral_radius = max(eigenvalues)
                spe_list.append(spectral_radius)
                # spectral_min = min(eigenvalues[eigenvalues > 0])

                # cluster_coefficient = nx.average_clustering(G)
                # clu_list.append(cluster_coefficient)

                density = nx.density(G)
                den_list.append(density)
                # sparsity = 1 - density

                homophily = calculate_homophily(label, ad)
                hom_list.append(homophily)
                db_index = davies_bouldin_score(feat, label)
                db_list.append(db_index)
                ad = normalize_adj(ad)

                db_index_agg = davies_bouldin_score(ad @ ad @ feat, label)
                db_agg_list.append(db_index_agg)
            spe_list = np.array(spe_list)
            eigtrace_list = np.array(eigtrace_list)
            clu_list = np.array(clu_list)
            den_list = np.array(den_list)
            hom_list = np.array(hom_list)
            db_list = np.array(db_list)
            db_agg_list = np.array(db_agg_list)
            args.logger.info(f"Average Density %: {np.mean(den_list) * 100}")
            args.logger.info(f"Average LapSpaceTrace: {np.mean(eigtrace_list)}")
            args.logger.info(f"Average Spectral Radius: {np.mean(spe_list)}")
            # args.logger.info(f"Average Cluster Coefficient: {np.mean(clu_list)}")
            args.logger.info(f"Average Homophily: {np.mean(hom_list)}")
            args.logger.info(f"Average Davies-Bouldin Index: {np.mean(db_list)}")
            args.logger.info(f"Average Davies-Bouldin Index AGG: {np.mean(db_agg_list)}")


        else:
            G = nx.from_numpy_array(adj)

            laplacian_matrix = nx.laplacian_matrix(G).astype(np.float32)

            # Compute the largest eigenvalue using sparse linear algebra
            k = 1  # number of eigenvalues and eigenvectors to compute
            eigenvalues, eigenvector = eigsh(laplacian_matrix, k=k, which='LM')
            eig = feat.T @ eigenvector @ eigenvector.T @ feat
            trace = np.trace(eig) / feat.shape[1]

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
            args.logger.info(f"LapSpaceTrace: {trace}")
            args.logger.info(f"Spectral Radius: {spectral_radius}")
            # print("Spectral Min:", spectral_min)
            # args.logger.info(f"Cluster Coefficient: {cluster_coefficient}")
            # print("Density:", density)
            args.logger.info(f"Homophily: {homophily}")
            args.logger.info(f"Davies-Bouldin Index: {db_index}")
            args.logger.info(f"Davies-Bouldin Index AGG: {db_index_agg}")

    def graph_property_no_structure(self):
        args = self.args
        adj, feat, label = self.adj, self.feat, self.label
        db_list = []
        if len(adj.shape) == 3:
            for i in range(adj.shape[0]):
                db_index = davies_bouldin_score(feat, label)
                db_list.append(db_index)

            db_list = np.array(db_list)
            args.logger.info(f"Average Davies-Bouldin Index: {np.mean(db_list)}")

        else:
            db_index = davies_bouldin_score(feat, label)
            args.logger.info(f"Davies-Bouldin Index: {db_index}")

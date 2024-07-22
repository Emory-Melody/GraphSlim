import numpy as np
import torch
from torch_sparse import matmul

from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.sparsification.coreset_base import CoreSet
from graphslim.utils import normalize_adj_tensor, to_tensor
from graphslim.dataset import *
from graphslim.dataset.convertor import networkit_to_pyg, pyg_to_networkit


class EdgeSparsifier:
    def __init__(self, setting, data, args, **kwargs):
        self.args = args

    @verbose_time_memory
    def reduce(self, data, verbose=False, save=True):
        # differ from vertex sparsification, edge sparsification is conducted on the whole graph
        graph = pyg_to_networkit(data)

        args = self.args
        # TODO: support edge weight
        new_edge_list, new_edge_attr = self.edge_cutter(graph)

        data.adj_syn = ei2csr(new_edge_list, graph.numberOfNodes())
        if verbose:
            print('selected edges:', data.adj_syn.sum())

        data.adj_syn = to_tensor(data.adj_syn, device='cpu')
        if save:
            save_reduced(adj_syn=data.adj_syn, args=args)

        return data

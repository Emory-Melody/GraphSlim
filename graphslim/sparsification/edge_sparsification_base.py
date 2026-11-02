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
    """Base for pruning graph connectivity while retaining every input node.

    """
    def __init__(self, setting, data, args, **kwargs):
        """Initialize EdgeSparsifier and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        self.args = args

    @verbose_time_memory
    def reduce(self, data, verbose=False, save=True):
        # differ from vertex sparsification, edge sparsification is conducted on the whole graph
        """Keep every node of the evaluated graph and drop edges.

        Edge sparsification does not select nodes, so ``reduction_rate`` is a fraction of
        edges (``edge_reduction=True``). The sparsified adjacency must match the features
        the evaluator trains on: the full graph in the transductive setting and the
        training subgraph in the inductive setting.

        :param data: Graph data providing full-graph edges or an inductive
            training adjacency, according to ``args.setting``.
        :param verbose: Whether to print retained and original edge counts.
            Defaults to ``False``.
        :param save: Whether to save the synthetic adjacency checkpoint.
            Defaults to ``True``.
        :returns: The input data with ``adj_syn``, ``reduced_edges`` and
            ``reduction_metadata`` populated.
        :raises ValueError: If ``args.edge_reduction`` is disabled.
        """
        graph = pyg_to_networkit(data)

        args = self.args
        # TODO: support edge weight
        new_edge_list, new_edge_attr = self.edge_cutter(graph)

        data.adj_syn = ei2csr(new_edge_list, graph.numberOfNodes())
        if verbose:
            print("selected edges:", data.adj_syn.sum())

        data.adj_syn = to_tensor(data.adj_syn, device="cpu")
        if save:
            save_reduced(adj_syn=data.adj_syn, args=args)

        return data

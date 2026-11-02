import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit
import copy


class TSpanner(EdgeSparsifier):
    """Edge reducer constructing a random spanning subgraph with bounded stretch.

    """
    def __init__(self, setting, data, args, **kwargs):
        """Initialize TSpanner and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super(TSpanner, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        """Build a random spanning subgraph with stretch ``args.ts``; the reduction rate is unused.

        :param G: PyGSP graph to coarsen, or NetworKit graph for an edge sparsifier.
        :returns: Tuple ``(edge_index, edge_attributes)`` from the sparsified NetworKit graph.
        """
        args = self.args
        new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted())

        G_copy = copy.deepcopy(G)
        while G_copy.numberOfEdges():
            edge = nk.graphtools.randomEdge(G_copy)
            G_copy.removeEdge(*edge)
            if (
                nk.distance.BidirectionalDijkstra(G_copy, edge[0], edge[1]).run().getDistance()
                > args.ts
            ):
                new_G.addEdge(*edge)
        if args.verbose:
            nk.overview(new_G)
        edge_index, edge_attr = from_networkit(new_G)

        return edge_index, edge_attr

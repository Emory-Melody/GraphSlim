import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit
import copy


class TSpanner(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(TSpanner, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args
        new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted())


        G_copy = copy.deepcopy(G)
        while G_copy.numberOfEdges():
            edge = nk.graphtools.randomEdge(G_copy)
            G_copy.removeEdge(*edge)
            if nk.distance.BidirectionalDijkstra(G_copy, edge[0], edge[1]).run().getDistance() > args.ts:
                new_G.addEdge(*edge)
        if args.verbose:
            nk.overview(new_G)
        edge_index, edge_attr = from_networkit(new_G)

        return edge_index, edge_attr

import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class RandomEdge(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(RandomEdge, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args

        randomEdgeSparsifier = nk.sparsification.RandomEdgeSparsifier()
        randomGraph = randomEdgeSparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)
        if args.verbose:
            nk.overview(randomGraph)
        edge_index, edge_attr = from_networkit(randomGraph)

        return edge_index, edge_attr

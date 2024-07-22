import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class SpanningForest(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(SpanningForest, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args
        graph=nk.graph.SpanningForest(G).run().getForest()
        if args.verbose:
            nk.overview(graph)
        edge_index, edge_attr = from_networkit(graph)

        return edge_index, edge_attr

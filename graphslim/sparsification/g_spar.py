import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class GSpar(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(GSpar, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args

        jaccardSimilaritySparsifier = nk.sparsification.JaccardSimilaritySparsifier()
        Graph = jaccardSimilaritySparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)
        if args.verbose:
            nk.overview(Graph)
        edge_index, edge_attr = from_networkit(Graph)

        return edge_index, edge_attr

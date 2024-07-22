import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class Scan(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(Scan, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args

        scanSparsifier = nk.sparsification.SCANSparsifier()
        scanGraph = scanSparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)
        if args.verbose:
            nk.overview(scanGraph)
        edge_index, edge_attr = from_networkit(scanGraph)

        return edge_index, edge_attr

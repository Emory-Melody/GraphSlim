import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class Scan(EdgeSparsifier):
    """Edge sparsifier based on structural neighborhood similarity.

    """
    def __init__(self, setting, data, args, **kwargs):
        """Initialize Scan and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super(Scan, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        """Sparsify connectivity using NetworKit SCAN structural similarity scores.

        :param G: PyGSP graph to coarsen, or NetworKit graph for an edge sparsifier.
        :returns: Tuple ``(edge_index, edge_attributes)`` from the sparsified NetworKit graph.
        """
        args = self.args

        scanSparsifier = nk.sparsification.SCANSparsifier()
        scanGraph = scanSparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)
        if args.verbose:
            nk.overview(scanGraph)
        edge_index, edge_attr = from_networkit(scanGraph)

        return edge_index, edge_attr

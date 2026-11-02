import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class LocalDegree(EdgeSparsifier):
    """Edge sparsifier based on local degree scores.

    """
    def __init__(self, setting, data, args, **kwargs):
        """Initialize LocalDegree and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super(LocalDegree, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        """Retain edges favored by local degree scores at the configured edge fraction.

        :param G: PyGSP graph to coarsen, or NetworKit graph for an edge sparsifier.
        :returns: Tuple ``(edge_index, edge_attributes)`` from the sparsified NetworKit graph.
        """
        args = self.args

        EdgeSparsifier = nk.sparsification.LocalDegreeSparsifier()
        Graph = EdgeSparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)
        if args.verbose:
            nk.overview(Graph)
        edge_index, edge_attr = from_networkit(Graph)

        return edge_index, edge_attr

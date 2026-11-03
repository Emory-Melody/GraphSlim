import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class SpanningForest(EdgeSparsifier):
    """Connectivity-preserving spanning forest edge reducer."""

    def __init__(self, setting, data, args, **kwargs):
        """Initialize SpanningForest and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super(SpanningForest, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        """Keep a spanning forest; this method ignores the configured reduction rate.

        :param G: PyGSP graph to coarsen, or NetworKit graph for an edge sparsifier.
        :returns: Tuple ``(edge_index, edge_attributes)`` from the sparsified NetworKit graph.
        """
        args = self.args
        graph = nk.graph.SpanningForest(G).run().getForest()
        if args.verbose:
            nk.overview(graph)
        edge_index, edge_attr = from_networkit(graph)

        return edge_index, edge_attr

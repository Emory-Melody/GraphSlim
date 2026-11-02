import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class GSpar(EdgeSparsifier):
    """G-Spar edge sparsification reducer.

    """
    def __init__(self, setting, data, args, **kwargs):
        """Initialize GSpar and its configuration.

        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        super(GSpar, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        """Sparsify edges using NetworKit Jaccard similarity scores and the configured retained-edge fraction.

        :param G: NetworKit input graph.
        :returns: Tuple ``(edge_index, edge_attributes)`` from the sparsified NetworKit graph.
        """
        args = self.args

        jaccardSimilaritySparsifier = nk.sparsification.JaccardSimilaritySparsifier()
        Graph = jaccardSimilaritySparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)
        if args.verbose:
            nk.overview(Graph)
        edge_index, edge_attr = from_networkit(Graph)

        return edge_index, edge_attr

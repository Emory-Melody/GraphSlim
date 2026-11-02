from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class RandomEdge(NetworkitSparsifier):
    """Uniform random edge sparsification."""

    networkit_class = "RandomEdgeSparsifier"

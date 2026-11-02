from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class GSpar(NetworkitSparsifier):
    """G-Spar using NetworKit Jaccard similarity scores."""

    networkit_class = "JaccardSimilaritySparsifier"

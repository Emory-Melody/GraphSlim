from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class LocalDegree(NetworkitSparsifier):
    """Local-degree edge sparsification."""

    networkit_class = "LocalDegreeSparsifier"

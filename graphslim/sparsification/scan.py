from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class Scan(NetworkitSparsifier):
    """Edge sparsification by SCAN structural similarity."""

    networkit_class = "SCANSparsifier"

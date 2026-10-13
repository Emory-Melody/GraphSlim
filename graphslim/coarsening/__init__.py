"""Public coarsening API, imported on demand."""

from graphslim._lazy import lazy_exports

__all__, __getattr__, __dir__ = lazy_exports(
    globals(),
    {
        "NOPE": (".nope", "NOPE"),
        "Coarsen": (".coarsening_base", "Coarsen"),
        "VNG": (".vng", "VNG"),
        "Cluster": (".clustering", "Cluster"),
        "ClusterAgg": (".clusteringagg", "ClusterAgg"),
        "Average": (".averaging", "Average"),
        "VariationNeighborhoods": (".variation_neighborhoods", "VariationNeighborhoods"),
        "VariationEdges": (".variation_edges", "VariationEdges"),
        "VariationCliques": (".variation_cliques", "VariationCliques"),
        "HeavyEdge": (".heavy_edge", "HeavyEdge"),
        "AlgebraicJc": (".algebraic_jc", "AlgebraicJc"),
        "AffinityGs": (".affinity_gs", "AffinityGs"),
        "Kron": (".kron", "Kron"),
    },
)

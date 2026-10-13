"""Public sparsification API, imported on demand."""

from graphslim._lazy import lazy_exports

__all__, __getattr__, __dir__ = lazy_exports(
    globals(),
    {
        "Herding": (".herding", "Herding"),
        "KCenter": (".kcenter", "KCenter"),
        "Random": (".random", "Random"),
        "CentP": (".cent_pagerank", "CentP"),
        "CentD": (".cent_degree", "CentD"),
        "KCenterAgg": (".kcenter_agg", "KCenterAgg"),
        "HerdingAgg": (".herding_agg", "HerdingAgg"),
        "RandomEdge": (".random_edge", "RandomEdge"),
        "SpanningForest": (".spanning_forest", "SpanningForest"),
        "GSpar": (".g_spar", "GSpar"),
        "LocalDegree": (".local_degree", "LocalDegree"),
        "RankDegree": (".rank_degree", "RankDegree"),
        "Scan": (".scan", "Scan"),
        "TSpanner": (".t_spanner", "TSpanner"),
    },
)

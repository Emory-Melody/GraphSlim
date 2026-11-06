# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
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
        "SimmelianParametric": (".networkit_sparsifiers", "SimmelianParametric"),
        "SimmelianNonParametric": (".networkit_sparsifiers", "SimmelianNonParametric"),
        "QuadrilateralSimmelian": (".networkit_sparsifiers", "QuadrilateralSimmelian"),
        "SimmelianMultiscale": (".networkit_sparsifiers", "SimmelianMultiscale"),
        "DegreeMultiscale": (".networkit_sparsifiers", "DegreeMultiscale"),
        "LocalSimilarity": (".networkit_sparsifiers", "LocalSimilarity"),
        "Multiscale": (".networkit_sparsifiers", "Multiscale"),
        "RandomNodeEdge": (".networkit_sparsifiers", "RandomNodeEdge"),
        "ForestFire": (".networkit_sparsifiers", "ForestFire"),
        "Triangle": (".networkit_sparsifiers", "Triangle"),
        "AlgebraicDistance": (".networkit_sparsifiers", "AlgebraicDistance"),
    },
)

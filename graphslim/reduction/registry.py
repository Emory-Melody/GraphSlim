"""Shared registry for graph reduction methods.

The registry keeps orchestration code independent from concrete method classes.
Methods are imported lazily so importing :mod:`graphslim.reduction` stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, Optional, Type


@dataclass(frozen=True)
class MethodSpec:
    """Description of a graph reduction method."""

    name: str
    family: str
    module: str
    class_name: str
    agg_module: Optional[str] = None
    agg_class_name: Optional[str] = None
    # Edge sparsifiers keep every node and interpret ``reduction_rate`` as the
    # fraction of edges retained (see ``--edge_reduction``).
    edge_reduction: bool = False

    def load_class(self, use_agg: bool = False) -> Type:
        """Import and return the registered reducer class lazily.

        :param use_agg: Whether to prefer the registered feature aggregation variant. Defaults to ``False``.
        :returns: Reducer class, using the aggregation variant when requested and registered.
        """
        module_name = self.module
        class_name = self.class_name
        if use_agg and self.agg_module and self.agg_class_name:
            module_name = self.agg_module
            class_name = self.agg_class_name
        module = import_module(module_name)
        return getattr(module, class_name)


def normalize_method_name(method: str) -> str:
    """Strip whitespace, replace hyphens with underscores, and lowercase a method name.

    :param method: Reduction method name.
    :returns: Normalized method name string.
    """

    return method.strip().replace("-", "_").lower()


_METHODS: Dict[str, MethodSpec] = {
    # Sparsification: node/core-set methods.
    "kcenter": MethodSpec(
        "kcenter",
        "sparsification",
        "graphslim.sparsification.kcenter",
        "KCenter",
        "graphslim.sparsification.kcenter_agg",
        "KCenterAgg",
    ),
    "herding": MethodSpec(
        "herding",
        "sparsification",
        "graphslim.sparsification.herding",
        "Herding",
        "graphslim.sparsification.herding_agg",
        "HerdingAgg",
    ),
    "random": MethodSpec("random", "sparsification", "graphslim.sparsification.random", "Random"),
    "cent_p": MethodSpec(
        "cent_p", "sparsification", "graphslim.sparsification.cent_pagerank", "CentP"
    ),
    "cent_d": MethodSpec(
        "cent_d", "sparsification", "graphslim.sparsification.cent_degree", "CentD"
    ),
    # Sparsification: edge methods (every node is kept; the rate applies to edges).
    "random_edge": MethodSpec(
        "random_edge",
        "sparsification",
        "graphslim.sparsification.random_edge",
        "RandomEdge",
        edge_reduction=True,
    ),
    "spanning_forest": MethodSpec(
        "spanning_forest",
        "sparsification",
        "graphslim.sparsification.spanning_forest",
        "SpanningForest",
        edge_reduction=True,
    ),
    "g_spar": MethodSpec(
        "g_spar", "sparsification", "graphslim.sparsification.g_spar", "GSpar", edge_reduction=True
    ),
    "local_degree": MethodSpec(
        "local_degree",
        "sparsification",
        "graphslim.sparsification.local_degree",
        "LocalDegree",
        edge_reduction=True,
    ),
    "rank_degree": MethodSpec(
        "rank_degree",
        "sparsification",
        "graphslim.sparsification.rank_degree",
        "RankDegree",
        edge_reduction=True,
    ),
    "scan": MethodSpec(
        "scan", "sparsification", "graphslim.sparsification.scan", "Scan", edge_reduction=True
    ),
    "t_spanner": MethodSpec(
        "t_spanner",
        "sparsification",
        "graphslim.sparsification.t_spanner",
        "TSpanner",
        edge_reduction=True,
    ),
    # Remaining concrete sparsifiers from NetworKit's Python module.
    **{
        name: MethodSpec(
            name, "sparsification", "graphslim.sparsification.networkit_sparsifiers",
            class_name, edge_reduction=True,
        )
        for name, class_name in (
            ("simmelian_parametric", "SimmelianParametric"),
            ("simmelian_non_parametric", "SimmelianNonParametric"),
            ("quadrilateral_simmelian", "QuadrilateralSimmelian"),
            ("simmelian_multiscale", "SimmelianMultiscale"),
            ("degree_multiscale", "DegreeMultiscale"),
            ("local_similarity", "LocalSimilarity"),
            ("multiscale", "Multiscale"),
            ("random_node_edge", "RandomNodeEdge"),
            ("forest_fire", "ForestFire"),
            ("triangle", "Triangle"),
            ("algebraic_distance", "AlgebraicDistance"),
        )
    },
    # Coarsening methods.
    "vng": MethodSpec("vng", "coarsening", "graphslim.coarsening.vng", "VNG"),
    "clustering": MethodSpec(
        "clustering",
        "coarsening",
        "graphslim.coarsening.clustering",
        "Cluster",
        "graphslim.coarsening.clusteringagg",
        "ClusterAgg",
    ),
    "averaging": MethodSpec("averaging", "coarsening", "graphslim.coarsening.averaging", "Average"),
    "variation_neighborhoods": MethodSpec(
        "variation_neighborhoods",
        "coarsening",
        "graphslim.coarsening.variation_neighborhoods",
        "VariationNeighborhoods",
    ),
    "variation_edges": MethodSpec(
        "variation_edges", "coarsening", "graphslim.coarsening.variation_edges", "VariationEdges"
    ),
    "variation_cliques": MethodSpec(
        "variation_cliques",
        "coarsening",
        "graphslim.coarsening.variation_cliques",
        "VariationCliques",
    ),
    "heavy_edge": MethodSpec(
        "heavy_edge", "coarsening", "graphslim.coarsening.heavy_edge", "HeavyEdge"
    ),
    "algebraic_jc": MethodSpec(
        "algebraic_jc", "coarsening", "graphslim.coarsening.algebraic_jc", "AlgebraicJc"
    ),
    "affinity_gs": MethodSpec(
        "affinity_gs", "coarsening", "graphslim.coarsening.affinity_gs", "AffinityGs"
    ),
    "kron": MethodSpec("kron", "coarsening", "graphslim.coarsening.kron", "Kron"),
    # Condensation methods.
    "cgc": MethodSpec("cgc", "condensation", "graphslim.condensation.cgc", "CGC"),
    "clustgdd": MethodSpec(
        "clustgdd", "condensation", "graphslim.condensation.clustgdd", "ClustGDD"
    ),
    "gcgp": MethodSpec("gcgp", "condensation", "graphslim.condensation.gcgp", "GCGP"),
    "gcond": MethodSpec("gcond", "condensation", "graphslim.condensation.gcond", "GCond"),
    "doscond": MethodSpec("doscond", "condensation", "graphslim.condensation.doscond", "DosCond"),
    "doscondx": MethodSpec(
        "doscondx", "condensation", "graphslim.condensation.doscondx", "DosCondX"
    ),
    "sfgc": MethodSpec("sfgc", "condensation", "graphslim.condensation.sfgc", "SFGC"),
    "sgdd": MethodSpec("sgdd", "condensation", "graphslim.condensation.sgdd", "SGDD"),
    "gcsntk": MethodSpec("gcsntk", "condensation", "graphslim.condensation.gcsntk", "GCSNTK"),
    "msgc": MethodSpec("msgc", "condensation", "graphslim.condensation.msgc", "MSGC"),
    "geom": MethodSpec("geom", "condensation", "graphslim.condensation.geom", "GEOM"),
    "simgc": MethodSpec("simgc", "condensation", "graphslim.condensation.simgc", "SimGC"),
    "gcdm": MethodSpec("gcdm", "condensation", "graphslim.condensation.gcdm", "GCDM"),
    "gcdmx": MethodSpec("gcdmx", "condensation", "graphslim.condensation.gcdmx", "GCDMX"),
    "gdem": MethodSpec("gdem", "condensation", "graphslim.condensation.gdem", "GDEM"),
    "gecc": MethodSpec("gecc", "condensation", "graphslim.condensation.gecc", "GECC"),
}

_ALIASES = {
    "algebraic_JC": "algebraic_jc",
    "affinity_GS": "affinity_gs",
    "tspanner": "t_spanner",
    "jaccard_similarity": "g_spar",
}


def get_method_spec(method: str) -> MethodSpec:
    """Resolve a method name or supported alias to its registry entry.

    Raises ``ValueError`` for an unknown method.

    :param method: Reduction method name.
    :returns: Immutable ``MethodSpec`` describing the canonical method.
    """
    key = normalize_method_name(method)
    key = _ALIASES.get(method, _ALIASES.get(key, key))
    try:
        return _METHODS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_METHODS))
        raise ValueError(
            f"Unknown graph reduction method '{method}'. Available methods: {choices}"
        ) from exc


def list_methods(family: Optional[str] = None) -> Iterable[str]:
    """List canonical reducer names in alphabetical order.

    :param family: Optional family filter: ``sparsification``, ``coarsening``, or ``condensation``. Defaults to ``None``.
    :returns: Tuple of registered names, optionally restricted to one family.
    """
    if family is None:
        return tuple(sorted(_METHODS))
    return tuple(sorted(name for name, spec in _METHODS.items() if spec.family == family))


def create_reducer(method: str, setting, data, args, **kwargs):
    """Resolve a registered method and instantiate its reducer.

    When ``args.agg`` is enabled, a registered aggregation variant is used
    if available. Unknown names raise ``ValueError``.

    :param method: Reduction method name.
    :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs.
    :param data: Graph data containing features, adjacency, labels, and split indices.
    :param args: Configuration namespace containing the options used by this operation.
    :param kwargs: Additional keyword options passed to the underlying implementation.
    :returns: Reducer instance constructed with ``setting``, ``data``, ``args``, and additional options.
    """

    spec = get_method_spec(method)
    use_agg = bool(getattr(args, "agg", False))
    cls = spec.load_class(use_agg=use_agg)
    return cls(setting=setting, data=data, args=args, **kwargs)

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

    def load_class(self, use_agg: bool = False) -> Type:
        module_name = self.module
        class_name = self.class_name
        if use_agg and self.agg_module and self.agg_class_name:
            module_name = self.agg_module
            class_name = self.agg_class_name
        module = import_module(module_name)
        return getattr(module, class_name)


def normalize_method_name(method: str) -> str:
    """Normalize method names while preserving public CLI aliases."""

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
    "cent_p": MethodSpec("cent_p", "sparsification", "graphslim.sparsification.cent_pagerank", "CentP"),
    "cent_d": MethodSpec("cent_d", "sparsification", "graphslim.sparsification.cent_degree", "CentD"),
    # Sparsification: edge methods.
    "random_edge": MethodSpec("random_edge", "sparsification", "graphslim.sparsification.random_edge", "RandomEdge"),
    "spanning_forest": MethodSpec(
        "spanning_forest", "sparsification", "graphslim.sparsification.spanning_forest", "SpanningForest"
    ),
    "g_spar": MethodSpec("g_spar", "sparsification", "graphslim.sparsification.g_spar", "GSpar"),
    "local_degree": MethodSpec("local_degree", "sparsification", "graphslim.sparsification.local_degree", "LocalDegree"),
    "rank_degree": MethodSpec("rank_degree", "sparsification", "graphslim.sparsification.rank_degree", "RankDegree"),
    "scan": MethodSpec("scan", "sparsification", "graphslim.sparsification.scan", "Scan"),
    "t_spanner": MethodSpec("t_spanner", "sparsification", "graphslim.sparsification.t_spanner", "TSpanner"),
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
    "variation_edges": MethodSpec("variation_edges", "coarsening", "graphslim.coarsening.variation_edges", "VariationEdges"),
    "variation_cliques": MethodSpec(
        "variation_cliques", "coarsening", "graphslim.coarsening.variation_cliques", "VariationCliques"
    ),
    "heavy_edge": MethodSpec("heavy_edge", "coarsening", "graphslim.coarsening.heavy_edge", "HeavyEdge"),
    "algebraic_jc": MethodSpec("algebraic_jc", "coarsening", "graphslim.coarsening.algebraic_jc", "AlgebraicJc"),
    "affinity_gs": MethodSpec("affinity_gs", "coarsening", "graphslim.coarsening.affinity_gs", "AffinityGs"),
    "kron": MethodSpec("kron", "coarsening", "graphslim.coarsening.kron", "Kron"),
    # Condensation methods.
    "gcond": MethodSpec("gcond", "condensation", "graphslim.condensation.gcond", "GCond"),
    "doscond": MethodSpec("doscond", "condensation", "graphslim.condensation.doscond", "DosCond"),
    "doscondx": MethodSpec("doscondx", "condensation", "graphslim.condensation.doscondx", "DosCondX"),
    "gcondx": MethodSpec("gcondx", "condensation", "graphslim.condensation.gcondx", "GCondX"),
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
}


def get_method_spec(method: str) -> MethodSpec:
    key = _ALIASES.get(method, normalize_method_name(method))
    try:
        return _METHODS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown graph reduction method '{method}'. Available methods: {choices}") from exc


def list_methods(family: Optional[str] = None) -> Iterable[str]:
    if family is None:
        return tuple(sorted(_METHODS))
    return tuple(sorted(name for name, spec in _METHODS.items() if spec.family == family))


def create_reducer(method: str, setting, data, args, **kwargs):
    """Instantiate a graph reduction method from the shared registry."""

    spec = get_method_spec(method)
    use_agg = bool(getattr(args, "agg", False))
    cls = spec.load_class(use_agg=use_agg)
    return cls(setting=setting, data=data, args=args, **kwargs)

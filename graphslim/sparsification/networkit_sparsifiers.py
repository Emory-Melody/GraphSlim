"""Adapters for NetworKit's concrete Python sparsifiers.

Native parameterization targets an edge fraction; tied scores can prevent an
exact hit. Triangle has no native size parameterization, so its adapter keeps
the highest scoring ``floor(rate * edges)`` edges, breaking ties by endpoints.
"""

import inspect
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import networkit as nk
import torch
from torch_geometric.utils.convert import from_networkit

from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier


class NetworkitSparsifier(EdgeSparsifier):
    """Run a native sparsifier with options from ``args.sparsifier_options``.

    Options use NetworKit's constructor names. ``local=True`` composes the
    generic LocalSparsifier wrapper without registering another method.
    DegreeMultiscale accepts ``max`` (default), ``min``, ``sum``, ``product``,
    or a Python callable as ``degsToAttrValue``.
    """

    networkit_class = None
    default_options = {}

    def __init__(self, setting, data, args, **kwargs):
        super().__init__(setting, data, args, **kwargs)
        supplied = getattr(args, "sparsifier_options", None)
        if isinstance(supplied, str):
            try:
                supplied = json.loads(supplied)
            except json.JSONDecodeError as exc:
                raise ValueError("sparsifier_options must be a JSON object") from exc
        if supplied is None:
            supplied = {}
        elif not isinstance(supplied, Mapping) and hasattr(supplied, "__dict__"):
            supplied = vars(supplied)
        if not isinstance(supplied, Mapping):
            raise ValueError("sparsifier_options must be a mapping or JSON object")
        options = {**self.default_options, **supplied}
        self.local = options.pop("local", False)
        if not isinstance(self.local, bool):
            raise ValueError("local must be a boolean")
        constructor = getattr(nk.sparsification, self.networkit_class)
        allowed = inspect.signature(constructor).parameters
        unknown = options.keys() - allowed.keys()
        if unknown:
            raise ValueError(
                f"Unknown {self.networkit_class} options: {', '.join(sorted(unknown))}"
            )
        for key in ("maxRank", "numberSystems", "numberIterations"):
            if key in options:
                value = options[key]
                if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
        for key in ("burnProbability", "omega", "targetBurntRatio"):
            if key in options:
                value = options[key]
                inclusive = key == "targetBurntRatio"
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, Real)
                    and math.isfinite(value)
                    and value > 0
                    and (value <= 1 if inclusive else value < 1)
                )
                if not valid:
                    interval = "(0, 1]" if inclusive else "(0, 1)"
                    raise ValueError(f"{key} must be in {interval}")
        if "above" in options and not isinstance(options["above"], bool):
            raise ValueError("above must be a boolean")
        if "norm" in options and (
            isinstance(options["norm"], bool) or options["norm"] not in (0, 2)
        ):
            raise ValueError("norm must be 0 (maximum) or 2 (Euclidean)")
        self.options = options.copy()
        if "degsToAttrValue" in options:
            mapper = options["degsToAttrValue"]
            if not callable(mapper):
                mappers = {
                    "max": max,
                    "min": min,
                    "sum": lambda u, v: u + v,
                    "product": lambda u, v: u * v,
                }
                if not isinstance(mapper, str) or mapper not in mappers:
                    raise ValueError("degsToAttrValue must be max, min, sum, product, or callable")
                options["degsToAttrValue"] = mappers[mapper]
        self.sparsifier = constructor(**options)
        if self.local:
            self.sparsifier = nk.sparsification.LocalSparsifier(self.sparsifier)

    def edge_cutter(self, G):
        """Return selected input edges, retaining the original node numbering."""
        rate = self.args.reduction_rate
        if (
            isinstance(rate, bool)
            or not isinstance(rate, Real)
            or not math.isfinite(rate)
            or not 0 <= rate <= 1
        ):
            raise ValueError("An edge reduction rate must be between 0 and 1")
        if G.isDirected():
            raise ValueError("NetworKit sparsifiers require an undirected input graph")
        seed = getattr(self.args, "seed", None)
        if seed is not None:
            nk.setSeed(int(seed), True)
        G.indexEdges()
        triangle_budget = self.networkit_class == "TriangleSparsifier" and not self.local
        self.sparsifier_metadata = {
            "networkit_class": self.networkit_class,
            "networkit_version": nk.__version__,
            "requested_edge_fraction": float(rate),
            "budget_selection": "triangle_top_k_endpoint_ties"
            if triangle_budget
            else "native_parameterization",
            "local_filter": self.local,
            "networkit_seed": None if seed is None else int(seed),
        }
        if rate == 0 or G.numberOfEdges() == 0:
            return torch.empty((2, 0), dtype=torch.long), None
        if rate == 1:
            sparse_graph = G
        elif triangle_budget:
            scores = self.sparsifier.scores(G)
            ranked = sorted(
                G.iterEdges(), key=lambda uv: (-scores[G.edgeId(*uv)], min(uv), max(uv))
            )
            sparse_graph = nk.Graph(G.upperNodeIdBound(), weighted=G.isWeighted())
            for u, v in ranked[: int(rate * G.numberOfEdges())]:
                sparse_graph.addEdge(u, v, G.weight(u, v))
        else:
            sparse_graph = self.sparsifier.getSparsifiedGraphOfSize(G, rate)
        if getattr(self.args, "verbose", False):
            nk.overview(sparse_graph)
        if sparse_graph.numberOfEdges() == 0:
            return torch.empty((2, 0), dtype=torch.long), None
        return from_networkit(sparse_graph)


class SimmelianParametric(NetworkitSparsifier):
    """Parametric Simmelian backbone; ``maxRank`` defaults to 10."""

    networkit_class = "SimmelianSparsifierParametric"
    default_options = {"maxRank": 10}


class SimmelianNonParametric(NetworkitSparsifier):
    """Nonparametric Simmelian backbone."""

    networkit_class = "SimmelianSparsifierNonParametric"


class QuadrilateralSimmelian(NetworkitSparsifier):
    """Simmelian backbone based on quadrangle overlap."""

    networkit_class = "QuadrilateralSimmelianSparsifier"


class SimmelianMultiscale(NetworkitSparsifier):
    """Multiscale backbone using triangle counts as weights."""

    networkit_class = "SimmelianMultiscaleSparsifier"


class DegreeMultiscale(NetworkitSparsifier):
    """Degree-weighted multiscale backbone; degree mapping defaults to max."""

    networkit_class = "DegreeMultiscaleSparsifier"
    default_options = {"degsToAttrValue": "max"}


class LocalSimilarity(NetworkitSparsifier):
    """Local similarity sparsification."""

    networkit_class = "LocalSimilaritySparsifier"


class Multiscale(NetworkitSparsifier):
    """Multiscale backbone of the input graph's edge weights."""

    networkit_class = "MultiscaleSparsifier"


class RandomNodeEdge(NetworkitSparsifier):
    """Random node-edge scores; output preserves every input node ID."""

    networkit_class = "RandomNodeEdgeSparsifier"


class ForestFire(NetworkitSparsifier):
    """Forest Fire scores, default burnProbability=0.5, targetBurntRatio=1."""

    networkit_class = "ForestFireSparsifier"
    default_options = {"burnProbability": 0.5, "targetBurntRatio": 1.0}


class Triangle(NetworkitSparsifier):
    """Triangle-count ranking with an exact floor(rate * edges) budget."""

    networkit_class = "TriangleSparsifier"


class AlgebraicDistance(NetworkitSparsifier):
    """Inverted algebraic-distance scores with NetworKit constructor defaults."""

    networkit_class = "AlgebraicDistanceSparsifier"

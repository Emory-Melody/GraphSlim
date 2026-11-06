# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Public models API, imported on demand."""

from graphslim._lazy import lazy_exports

DOWNSTREAM_MODELS = ("GCN", "SGC", "APPNP", "Cheby", "GraphSage", "GAT", "SGFormer", "GIN")


def get_model_class(name):
    """Resolve a supported downstream GNN architecture by name.

    :param name: Case-sensitive architecture name from ``DOWNSTREAM_MODELS``, such
        as ``'GCN'`` or ``'SGFormer'``.
    :returns: The requested model class, imported on demand.
    :raises ValueError: If the architecture name is unsupported.
    """
    supported = DOWNSTREAM_MODELS
    if name not in supported:
        raise ValueError(f"Unknown GNN model {name!r}; choose from {supported}")
    return globals().get(name) or __getattr__(name)


__all__, __getattr__, __dir__ = lazy_exports(
    globals(),
    {
        "APPNP": (".appnp", "APPNP"),
        "Cheby": (".cheby", "Cheby"),
        "GAT": (".gat", "GAT"),
        "GCN": (".gcn", "GCN"),
        "GIN": (".gin", "GIN"),
        "GraphSage": (".graphsage", "GraphSage"),
        "IGNR": (".ignr", "GraphonLearner"),
        "KernelRidgeRegression": (".krr", "KernelRidgeRegression"),
        "PGE": (".parametrized_adj", "PGE"),
        "SGC": (".sgc", "SGC"),
        "StructureBasedNeuralTangentKernel": (".sntk", "StructureBasedNeuralTangentKernel"),
        "PRBCD": (".prbcd", "PRBCD"),
        "RandomAttack": (".random_attack", "RandomAttack"),
        "SGFormer": (".sgformer", "SGFormer"),
    },
)

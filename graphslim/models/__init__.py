"""Public models API, imported on demand."""

from graphslim._lazy import lazy_exports

DOWNSTREAM_MODELS = ("GCN", "SGC", "APPNP", "Cheby", "GraphSage", "GAT", "SGFormer")


def get_model_class(name):
    """Resolve supported GNN architectures without evaluating Python code."""
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

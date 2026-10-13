"""Public evaluation API, imported on demand."""

from graphslim._lazy import lazy_exports

__all__, __getattr__, __dir__ = lazy_exports(
    globals(),
    {
        "Evaluator": (".eval_agent", "Evaluator"),
        "NasEvaluator": (".nas_eval", "NasEvaluator"),
        "PropertyEvaluator": (".graph_property", "PropertyEvaluator"),
        "evaluate": (".utils", "evaluate"),
        "verbose_time_memory": (".utils", "verbose_time_memory"),
    },
)

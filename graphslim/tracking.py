"""Optional experiment tracking for graph reduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def _compact_config(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    values = getattr(args, "__dict__", {})
    config = {}
    for key, value in values.items():
        if key.startswith("_") or key == "logger":
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            config[key] = value
    return config


@dataclass
class NullTracker:
    """No-op tracker used when WandB is disabled or unavailable."""

    enabled: bool = False

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        return None

    def log_graph(self, name: str, graph: Any, step: Optional[int] = None) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbTracker:
    """Thin wrapper around WandB with graph-specific metric names."""

    enabled = True

    def __init__(self, args: Any = None, project: Optional[str] = None, run_name: Optional[str] = None):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("WandB tracking requested but the 'wandb' package is not installed.") from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=project or getattr(args, "wandb_project", "graphslim"),
            name=run_name or getattr(args, "wandb_run_name", None),
            config=_compact_config(args),
        )

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        self._wandb.log(dict(metrics), step=step)

    def log_graph(self, name: str, graph: Any, step: Optional[int] = None) -> None:
        summary = graph_summary(graph)
        self.log({f"{name}/{key}": value for key, value in summary.items()}, step=step)

    def finish(self) -> None:
        self._wandb.finish()


def build_tracker(args: Any = None):
    """Create a WandB tracker only when explicitly requested."""

    if not bool(getattr(args, "wandb", False)):
        return NullTracker()
    try:
        return WandbTracker(args)
    except RuntimeError:
        if bool(getattr(args, "wandb_required", False)):
            raise
        return NullTracker()


def graph_summary(graph: Any) -> Dict[str, Any]:
    """Return cheap graph evolution metrics for PyG-like or reduced graph objects."""

    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None and hasattr(graph, "x"):
        num_nodes = graph.x.shape[0]
    if num_nodes is None and hasattr(graph, "feat_syn"):
        num_nodes = graph.feat_syn.shape[0]

    num_edges = None
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None:
        num_edges = int(edge_index.shape[1])
    elif hasattr(graph, "adj_syn"):
        adj = graph.adj_syn
        if getattr(adj, "is_sparse", False) and hasattr(adj, "_nnz"):
            num_edges = int(adj._nnz())
        elif hasattr(adj, "nnz"):
            num_edges = int(adj.nnz)
        elif hasattr(adj, "sum"):
            num_edges = int(adj.sum().item() if hasattr(adj.sum(), "item") else adj.sum())

    metrics = {}
    if num_nodes is not None:
        metrics["nodes"] = int(num_nodes)
    if num_edges is not None:
        metrics["edges"] = int(num_edges)
    if num_nodes and num_edges is not None:
        metrics["density"] = float(num_edges) / float(max(num_nodes * max(num_nodes - 1, 1), 1))
    return metrics

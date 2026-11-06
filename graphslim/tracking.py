# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Optional experiment tracking for graph reduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def _compact_config(args: Any) -> Dict[str, Any]:
    """Keep public scalar configuration attributes for experiment tracking.

    :param args: Configuration namespace containing the options used by this operation.
    :returns: Dictionary excluding loggers, private attributes, and non-scalar values.
    """
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
        """Accept metric logging calls when experiment tracking is disabled.

        :param metrics: Mapping from metric names to scalar values.
        :param step: Optional training step attached to these logged values. Defaults to ``None``.
        """
        return None

    def log_graph(self, name: str, graph: Any, step: Optional[int] = None) -> None:
        """Accept graph logging calls when experiment tracking is disabled.

        :param name: Prefix used to group the graph summary metrics.
        :param graph: Input graph object.
        :param step: Optional training step attached to these logged values. Defaults to ``None``.
        """
        return None

    def finish(self) -> None:
        """Close a disabled tracker without external side effects."""
        return None


class WandbTracker:
    """Thin wrapper around WandB with graph-specific metric names."""

    enabled = True

    def __init__(
        self, args: Any = None, project: Optional[str] = None, run_name: Optional[str] = None
    ):
        """Initialize WandbTracker and its configuration.

        :param args: Configuration namespace containing the options used by this operation. Defaults to ``None``.
        :param project: Weights & Biases project name; ``None`` uses configuration or defaults. Defaults to ``None``.
        :param run_name: Optional display name for the tracking run. Defaults to ``None``.
        """
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "WandB tracking requested but the 'wandb' package is not installed."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=project or getattr(args, "wandb_project", "graphslim"),
            name=run_name or getattr(args, "wandb_run_name", None),
            config=_compact_config(args),
        )

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Send scalar metrics to the active Weights & Biases run.

        :param metrics: Mapping from metric names to scalar values.
        :param step: Optional training step attached to these logged values. Defaults to ``None``.
        """
        self._wandb.log(dict(metrics), step=step)

    def log_graph(self, name: str, graph: Any, step: Optional[int] = None) -> None:
        """Summarize a graph and log its size and attribute statistics under a name prefix.

        :param name: Prefix used to group the graph summary metrics.
        :param graph: Input graph object.
        :param step: Optional training step attached to these logged values. Defaults to ``None``.
        """
        summary = graph_summary(graph)
        self.log({f"{name}/{key}": value for key, value in summary.items()}, step=step)

    def finish(self) -> None:
        """Finish the active Weights & Biases run."""
        self._wandb.finish()


def build_tracker(args: Any = None):
    """Create experiment tracking only when explicitly enabled in the configuration.

    :param args: Configuration namespace containing the options used by this operation. Defaults to ``None``.
    :returns: WandbTracker when enabled, otherwise NullTracker; optional initialization failures follow the required-tracking setting.
    """

    if not bool(getattr(args, "wandb", False)):
        return NullTracker()
    try:
        return WandbTracker(args)
    except RuntimeError:
        if bool(getattr(args, "wandb_required", False)):
            raise
        return NullTracker()


def graph_summary(graph: Any) -> Dict[str, Any]:
    """Summarize node, edge, feature, and class counts without expensive graph diagnostics.

    :param graph: Input graph object.
    :returns: Mapping of available graph size and attribute statistics.
    """

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
        # Count stored edges, never total edge weight.
        if str(getattr(adj, "layout", "")).startswith("torch.sparse") and hasattr(adj, "_nnz"):
            num_edges = int(adj._nnz())
        elif hasattr(adj, "nnz"):
            nnz = adj.nnz  # SciPy attribute or torch_sparse method
            num_edges = int(nnz() if callable(nnz) else nnz)
        elif hasattr(adj, "count_nonzero"):
            num_edges = int(adj.count_nonzero())
        else:
            import numpy as np

            num_edges = int(np.count_nonzero(adj))

    metrics = {}
    if num_nodes is not None:
        metrics["nodes"] = int(num_nodes)
    if num_edges is not None:
        metrics["edges"] = int(num_edges)
    if num_nodes and num_edges is not None:
        metrics["density"] = float(num_edges) / float(max(num_nodes * max(num_nodes - 1, 1), 1))
    return metrics

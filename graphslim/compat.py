"""Compatibility helpers for PyG and DGL graph objects."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch_geometric.data import Data


def is_pyg_data(graph: Any) -> bool:
    return isinstance(graph, Data)


def is_dgl_graph(graph: Any) -> bool:
    return graph.__class__.__module__.startswith("dgl.") or graph.__class__.__module__ == "dgl"


def to_pyg_data(graph: Any, *, node_feature: str = "feat", node_label: str = "label") -> Data:
    """Convert a DGL graph or PyG-like reduced object to PyG ``Data``."""

    if isinstance(graph, Data):
        return graph

    if is_dgl_graph(graph):
        row, col = graph.edges()
        data = Data(edge_index=torch.stack([row, col], dim=0), num_nodes=graph.num_nodes())
        if node_feature in graph.ndata:
            data.x = graph.ndata[node_feature]
        elif "feature" in graph.ndata:
            data.x = graph.ndata["feature"]
        if node_label in graph.ndata:
            data.y = graph.ndata[node_label]
        return data

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None and hasattr(graph, "adj_syn"):
        adj = graph.adj_syn
        if hasattr(adj, "coalesce"):
            edge_index = adj.coalesce().indices()
        elif hasattr(adj, "nonzero"):
            nz = adj.nonzero()
            edge_index = torch.as_tensor(nz if len(nz) == 2 else nz.T, dtype=torch.long)
    data = Data(edge_index=edge_index)
    if hasattr(graph, "feat_syn"):
        data.x = graph.feat_syn
    elif hasattr(graph, "x"):
        data.x = graph.x
    if hasattr(graph, "labels_syn"):
        data.y = graph.labels_syn
    elif hasattr(graph, "y"):
        data.y = graph.y
    if hasattr(data, "x") and data.x is not None:
        data.num_nodes = data.x.shape[0]
    return data


def to_dgl_graph(graph: Any, *, node_feature: str = "feat", node_label: str = "label"):
    """Convert a PyG graph to DGL when DGL is installed."""

    try:
        import dgl
    except ImportError as exc:
        raise RuntimeError("DGL conversion requested but the 'dgl' package is not installed.") from exc

    if is_dgl_graph(graph):
        return graph

    data = to_pyg_data(graph)
    edge_index = data.edge_index
    if edge_index is None:
        raise ValueError("Cannot convert graph without edge_index to DGL.")
    num_nodes: Optional[int] = getattr(data, "num_nodes", None)
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    if getattr(data, "x", None) is not None:
        g.ndata[node_feature] = data.x
    if getattr(data, "y", None) is not None:
        g.ndata[node_label] = data.y
    return g

"""Compatibility helpers for PyG and DGL graph objects."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch_geometric.data import Data


def is_pyg_data(graph: Any) -> bool:
    return isinstance(graph, Data)


def is_dgl_graph(graph: Any) -> bool:
    return graph.__class__.__module__.startswith("dgl.") or graph.__class__.__module__ == "dgl"


def to_pyg_data(
    graph: Any,
    *,
    node_feature: str = "feat",
    node_label: str = "label",
    edge_weight: str = "weight",
) -> Data:
    """Convert a DGL graph or PyG-like reduced object to PyG ``Data``."""

    if isinstance(graph, Data):
        return graph

    if is_dgl_graph(graph):
        if not graph.is_homogeneous:
            raise ValueError(
                "to_pyg_data expects a homogeneous DGL graph; convert typed relations explicitly."
            )
        row, col = graph.edges()
        data = Data(edge_index=torch.stack([row, col], dim=0), num_nodes=graph.num_nodes())
        if node_feature in graph.ndata:
            data.x = graph.ndata[node_feature]
        elif "feature" in graph.ndata:
            data.x = graph.ndata["feature"]
        if node_label in graph.ndata:
            data.y = graph.ndata[node_label]
        for mask in ("train_mask", "val_mask", "test_mask"):
            if mask in graph.ndata:
                data[mask] = graph.ndata[mask]
        for key in (edge_weight, "edge_weight"):
            if key in graph.edata:
                data.edge_weight = graph.edata[key]
                break
        if "edge_attr" in graph.edata:
            data.edge_attr = graph.edata["edge_attr"]
        return data

    edge_index = getattr(graph, "edge_index", None)
    weights = getattr(graph, "edge_weight", None)
    reduced = (
        getattr(graph, "feat_syn", None) is not None or getattr(graph, "adj_syn", None) is not None
    )
    if getattr(graph, "adj_syn", None) is not None:
        adj = graph.adj_syn
        if isinstance(adj, torch.Tensor):
            if adj.ndim != 2:
                raise ValueError("Convert one reduced adjacency matrix at a time.")
            if adj.layout == torch.sparse_coo:
                adj = adj.coalesce()
                edge_index, weights = adj.indices(), adj.values()
            elif adj.layout != torch.strided:
                adj = adj.to_sparse_coo().coalesce()
                edge_index, weights = adj.indices(), adj.values()
            else:
                edge_index = adj.nonzero(as_tuple=False).T.contiguous()
                weights = adj[edge_index[0], edge_index[1]]
        elif hasattr(adj, "coo"):
            row, col, weights = adj.coo()
            edge_index = torch.stack([row, col])
        elif hasattr(adj, "tocoo"):
            coo = adj.tocoo()
            import numpy as np

            edge_index = torch.from_numpy(np.stack((coo.row, coo.col))).long()
            weights = torch.from_numpy(coo.data)
        else:
            raise TypeError("Unsupported reduced adjacency type.")
    data = Data(edge_index=edge_index)
    if getattr(graph, "feat_syn", None) is not None:
        data.x = graph.feat_syn
    elif hasattr(graph, "x"):
        data.x = graph.x
    if getattr(graph, "labels_syn", None) is not None:
        data.y = graph.labels_syn
    elif hasattr(graph, "y"):
        data.y = graph.y
    if hasattr(data, "x") and data.x is not None:
        data.num_nodes = data.x.shape[0]
    elif getattr(graph, "num_nodes", None) is not None:
        data.num_nodes = graph.num_nodes
    if weights is not None:
        data.edge_weight = weights
    if not reduced:
        for key in ("train_mask", "val_mask", "test_mask", "edge_attr"):
            value = getattr(graph, key, None)
            if value is not None:
                data[key] = value
    return data


def to_dgl_graph(
    graph: Any,
    *,
    node_feature: str = "feat",
    node_label: str = "label",
    edge_weight: str = "weight",
):
    """Convert a PyG graph to DGL when DGL is installed."""

    try:
        import dgl
    except (ImportError, OSError, FileNotFoundError) as exc:
        raise RuntimeError(
            "DGL could not be imported. Install a DGL build compatible with this Torch/runtime; "
            "use python -m graphslim.doctor --dgl for diagnostics. DGL is optional."
        ) from exc

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
    for mask in ("train_mask", "val_mask", "test_mask"):
        if getattr(data, mask, None) is not None:
            g.ndata[mask] = data[mask]
    if getattr(data, "edge_weight", None) is not None:
        g.edata[edge_weight] = data.edge_weight
    if getattr(data, "edge_attr", None) is not None:
        g.edata["edge_attr"] = data.edge_attr
    return g

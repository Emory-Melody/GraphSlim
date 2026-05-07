"""Local visualization helpers for small original/reduced graphs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from graphslim.compat import to_pyg_data


def to_networkx_graph(graph: Any) -> nx.Graph:
    data = to_pyg_data(graph)
    if data.edge_index is not None:
        return to_networkx(data, to_undirected=True)
    if hasattr(data, "x") and data.x is not None:
        return nx.empty_graph(data.x.shape[0])
    return nx.Graph()


def draw_graph_pair(
    original: Any,
    reduced: Any,
    *,
    title: str = "GraphSlim reduction",
    output_path: Optional[str] = None,
    max_nodes: int = 300,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Draw original and reduced graphs side by side.

    This intentionally targets small graph inspection; large benchmark graphs
    should be sampled before plotting.
    """

    graphs = [to_networkx_graph(original), to_networkx_graph(reduced)]
    labels = ["Original", "Reduced"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    for ax, graph, label in zip(axes, graphs, labels):
        ax.set_title(f"{label}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        ax.axis("off")
        if graph.number_of_nodes() > max_nodes:
            sampled = list(graph.nodes())[:max_nodes]
            graph = graph.subgraph(sampled).copy()
        pos = nx.spring_layout(graph, seed=7) if graph.number_of_nodes() else {}
        nx.draw_networkx(graph, pos=pos, ax=ax, node_size=30, with_labels=False, width=0.4)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, axes


def load_reduced_graph(adj_path: str, feat_path: Optional[str] = None, label_path: Optional[str] = None):
    from torch_geometric.data import Data

    adj = torch.load(adj_path, map_location="cpu")
    if hasattr(adj, "coalesce"):
        edge_index = adj.coalesce().indices()
    elif hasattr(adj, "nonzero"):
        edge_index = torch.as_tensor(adj.nonzero(), dtype=torch.long)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
    else:
        raise ValueError(f"Unsupported adjacency format in {adj_path}")
    data = Data(edge_index=edge_index)
    if feat_path:
        data.x = torch.load(feat_path, map_location="cpu")
    if label_path:
        data.y = torch.load(label_path, map_location="cpu")
    return data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Visualize a small GraphSlim reduced graph.")
    parser.add_argument("--adj", required=True, help="path to saved reduced adjacency tensor")
    parser.add_argument("--feat", default=None, help="optional path to saved reduced features")
    parser.add_argument("--label", default=None, help="optional path to saved reduced labels")
    parser.add_argument("--out", default="graphslim_reduced.png", help="output image path")
    args = parser.parse_args(argv)

    reduced = load_reduced_graph(args.adj, args.feat, args.label)
    draw_graph_pair(reduced, reduced, title="GraphSlim reduced graph", output_path=args.out)


if __name__ == "__main__":
    main()

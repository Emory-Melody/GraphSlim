"""NOPE / NOPE*: non-selfish neighbor-interference graph coarsening.

Independent implementation of Bai et al., ICML 2026 (arXiv:2605.13021).
https://github.com/dazonglian/NOPE-main supplies the reference loss formulas.
No labels are used to choose merges; only training labels label supernodes.
"""

import heapq
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from graphslim.reduction.lightweight import LightweightReducer


def interference_loss(u, v, features, sizes, neighbors, relaxed=False, sum_sq=None):
    """Released NOPE loss, or the local-isotropy NOPE* surrogate."""
    weight = sizes[u] * sizes[v] / (sizes[u] + sizes[v] + 1e-8)
    common = neighbors[u] & neighbors[v]
    if relaxed:
        count = max(len(neighbors[u] | neighbors[v]) - 2, 0)
        return float(weight * count * np.square(features[u] - features[v]).sum())
    if sum_sq is None:
        sum_u = sum(float(np.dot(features[n], features[u])) ** 2 for n in neighbors[u])
        sum_v = sum(float(np.dot(features[n], features[v])) ** 2 for n in neighbors[v])
    else:
        sum_u, sum_v = sum_sq[u], sum_sq[v]
    difference = sum(float(np.dot(features[n], features[u] - features[v])) ** 2 for n in common)
    return float(weight * (sum_u + sum_v + difference))


def hierarchical_merge(features, adjacency, target, relaxed=False):
    """Heap-based edge contraction with versioned, refreshed neighbor costs.

    Never invent cross-component edges to hit an impossible target. The returned
    mapping includes isolated nodes and every original node exactly once.
    """
    features = np.asarray(features, dtype=np.float32)
    n, dim = features.shape
    vectors = np.zeros((2 * n, dim), dtype=np.float32)
    vectors[:n] = features
    sizes = np.ones(2 * n, dtype=np.float32)
    sum_sq = np.zeros(2 * n, dtype=np.float64)
    neighbors = {i: set() for i in range(n)}
    row, col = adjacency.nonzero()
    for u, v in zip(row, col):
        if u != v:
            neighbors[u].add(int(v))
            neighbors[v].add(int(u))
    edge_count = sum(map(len, neighbors.values())) // 2
    versions = np.zeros(2 * n, dtype=np.int64)
    active, tree = set(range(n)), {}

    def refresh(node):
        if not relaxed:
            indices = sorted(neighbors[node])
            sum_sq[node] = np.square(vectors[indices] @ vectors[node]).sum() if indices else 0.0
        versions[node] += 1

    heap = []

    def push(u, v):
        u, v = sorted((u, v))
        loss = interference_loss(u, v, vectors, sizes, neighbors, relaxed, sum_sq)
        heapq.heappush(heap, (loss, u, v, versions[u], versions[v]))

    for node in range(n):
        refresh(node)
    for u in range(n):
        for v in sorted(neighbors[u]):
            if u < v:
                push(u, v)
    next_id = n
    target = max(1, min(int(target), n))
    while len(active) > target and heap:
        _, u, v, version_u, version_v = heapq.heappop(heap)
        if u not in active or v not in active or v not in neighbors[u]:
            continue
        if versions[u] != version_u or versions[v] != version_v:
            continue
        joined = (neighbors[u] | neighbors[v]) - {u, v}
        # Remove the incident edges (u-v counted once), then add new-to-joined
        # edges. This avoids scanning every active node after every contraction.
        edge_count -= len(neighbors[u]) + len(neighbors[v]) - len(joined) - 1
        new = next_id
        next_id += 1
        sizes[new] = sizes[u] + sizes[v]
        vectors[new] = (sizes[u] * vectors[u] + sizes[v] * vectors[v]) / sizes[new]
        neighbors[new] = joined
        tree[new] = (u, v)
        active.difference_update((u, v))
        active.add(new)
        for neighbor in joined:
            neighbors[neighbor].difference_update((u, v))
            neighbors[neighbor].add(new)
        del neighbors[u], neighbors[v]
        affected = joined | {new}
        for node in affected:
            refresh(node)
        pairs = {
            tuple(sorted((node, neighbor))) for node in affected for neighbor in neighbors[node]
        }
        for a, b in sorted(pairs):
            push(a, b)
        # Every current edge already has one entry at its endpoint versions.
        # Drop stale entries without recomputing all unchanged edge losses.
        if len(heap) > max(1024, 4 * edge_count):
            heap[:] = [
                entry
                for entry in heap
                if entry[1] in active
                and entry[2] in active
                and entry[3] == versions[entry[1]]
                and entry[4] == versions[entry[2]]
                and entry[2] in neighbors[entry[1]]
            ]
            heapq.heapify(heap)
    roots = sorted(active)
    mapping = np.empty(n, dtype=np.int64)
    for cluster, root in enumerate(roots):
        stack = [root]
        while stack:
            node = stack.pop()
            if node in tree:
                stack.extend(tree[node])
            else:
                mapping[node] = cluster
    return mapping, vectors[roots], tree


class NOPE(LightweightReducer):
    def reduce(self, data, verbose=True, save=True):
        features, adjacency, train, labels = self.inputs(data)
        # Keep the package's training-relative rate convention, explicitly not
        # the author's whole-graph merge_ratio convention.
        budget_base = getattr(self.args, "nope_budget_base", "training")
        if budget_base not in ("training", "graph"):
            raise ValueError("nope_budget_base must be training or graph")
        base = len(train) if budget_base == "training" else len(features)
        target = max(1, round(base * self.args.reduction_rate))
        relaxed = getattr(self.args, "nope_relaxed", False)
        assignment, centers, _ = hierarchical_merge(
            features.cpu().numpy(), adjacency, target, relaxed
        )
        size = len(centers)
        train_clusters = assignment[train.cpu().numpy()]
        votes = np.zeros((size, data.nclass), dtype=np.int64)
        np.add.at(votes, (train_clusters, labels.cpu().numpy()), 1)
        labeled = np.flatnonzero(votes.sum(1))
        self.check_budget(len(labeled))
        membership = sp.csr_matrix(
            (np.ones(len(features)), (np.arange(len(features)), assignment)),
            shape=(len(features), size),
        )
        compressed = (membership.T @ adjacency @ membership).tocsr()
        compressed.setdiag(0)
        compressed.eliminate_zeros()
        compressed.data[:] = 1
        compressed = compressed.maximum(compressed.T)
        compressed = compressed[labeled][:, labeled]
        if size > target:
            warnings.warn(
                f"NOPE stopped at {size} components (requested {target}); disconnected components cannot be merged. Saved {len(labeled)} training-labeled supernodes.",
                RuntimeWarning,
                stacklevel=2,
            )
        data.nope_mapping = torch.from_numpy(assignment)
        return self.finish(
            data,
            torch.from_numpy(centers[labeled]),
            torch.from_numpy(compressed.toarray()).float(),
            torch.from_numpy(votes[labeled].argmax(1)).long(),
            save=save,
            verbose=verbose,
            gnn_training_steps=0,
            variant="NOPE*" if relaxed else "NOPE",
            budget_base=budget_base,
            requested_nodes=target,
            coarse_nodes=size,
            unlabeled_supernodes_removed=size - len(labeled),
            budget_reached=size <= target,
            label_policy="training-only majority; unlabeled supernodes excluded",
        )

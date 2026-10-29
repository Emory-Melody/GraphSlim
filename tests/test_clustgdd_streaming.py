"""Streaming must preserve global edge selection, including zero/tied weights."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from graphslim.condensation.clustgdd import compress_graph, iter_class_aware_views


def original_views(adjacency, embeddings, ratio):
    """Unchunked v3 reference, retained to detect scientific-protocol drift."""
    adjacency = adjacency.coalesce()
    src, dst = adjacency.indices()
    similarity = F.cosine_similarity(embeddings[src], embeddings[dst], dim=1).clamp_min(0)
    values = adjacency.values() * similarity
    degrees = torch.zeros(len(embeddings), device=embeddings.device).scatter_add_(0, src, values)
    resistance = values / degrees[src].clamp_min(1e-12) + values / degrees[dst].clamp_min(1e-12)
    probabilities = embeddings.softmax(1)
    size = int(len(values) * ratio)
    views = []
    for label in range(embeddings.shape[1]):
        weights = probabilities[src, label] * probabilities[dst, label] * resistance
        chosen = torch.topk(weights, size).indices
        indices = adjacency.indices()[:, chosen].cpu().numpy()
        graph = sp.csr_matrix((values[chosen].cpu().numpy(), indices), shape=adjacency.shape)
        views.append((graph + graph.T) * 0.5)
    return views


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ratio", [0, 0.4, 1])
@pytest.mark.parametrize("chunk_size", [1, 31, 100_000])
@pytest.mark.parametrize("tied", [False, True])
def test_streamed_views_and_compression_equal_reference(device, ratio, chunk_size, tied):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(512)
    n, classes = 37, 47
    edges = torch.randint(n, (2, 271), generator=generator)
    values = torch.rand(edges.shape[1], generator=generator)
    adjacency = torch.sparse_coo_tensor(edges, values, (n, n)).coalesce().to(device)
    embeddings = torch.randn(n, classes, generator=generator).to(device)
    if tied:
        embeddings.fill_(1)
        embeddings[::3] = 0
    assignments = np.arange(n) % 5
    expected = original_views(adjacency, embeddings, ratio)
    actual = iter_class_aware_views(adjacency, embeddings, ratio, chunk_size)
    assert iter(actual) is actual
    for left, right in zip(expected, actual, strict=True):
        assert (left != right).nnz == 0
        assert (
            compress_graph(left, assignments, 5) != compress_graph(right, assignments, 5)
        ).nnz == 0


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_reject_invalid_chunk_size(chunk_size):
    adjacency = torch.eye(3).to_sparse()
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_class_aware_views(adjacency, torch.ones(3, 2), 0.4, chunk_size))


@pytest.mark.parametrize("setting", ["trans", "ind"])
def test_full_reduction_matches_unchunked_reference(tmp_path, monkeypatch, setting):
    from test_lightweight_methods import quick_args, tiny_graph
    from graphslim.condensation import clustgdd
    from graphslim.utils import seed_everything

    args = quick_args(tmp_path, "clustgdd", setting)
    args.clustgdd_edge_chunk_size = 7
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seed_everything(19)
        graph = tiny_graph()
        actual = clustgdd.ClustGDD(setting, graph, args).reduce(graph, verbose=False, save=False)
        monkeypatch.setattr(
            clustgdd,
            "iter_class_aware_views",
            lambda adjacency, embeddings, ratio, chunk: iter(
                original_views(adjacency, embeddings, ratio)
            ),
        )
        seed_everything(19)
        graph = tiny_graph()
        expected = clustgdd.ClustGDD(setting, graph, args).reduce(graph, verbose=False, save=False)
        for name in ("feat_syn", "adj_syn", "labels_syn"):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
        for name in ("pretraining_validation", "refinement_validation", "final_train_loss"):
            assert actual.reduction_metadata[name] == expected.reduction_metadata[name]
    finally:
        torch.set_num_threads(previous_threads)

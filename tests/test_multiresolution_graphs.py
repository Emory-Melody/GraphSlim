# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Schur-complement and Laplacian invariants for multiresolution graphs."""

import numpy as np
import pytest
import scipy.sparse as sp
from pygsp import graphs

from graphslim.coarsening import utils


@pytest.mark.parametrize("full_eigen", [False, True])
def test_kron_pyramid_matches_schur_complement_and_keeps_original_node_mapping(full_eigen):
    graph = graphs.Path(32)
    graph.set_coordinates(np.column_stack([np.arange(32), np.zeros(32)]))
    pyramid = utils.my_graph_multiresolution(
        graph, 2, r=0.75, sparsify=False, compute_full_eigen=full_eigen
    )
    assert [level.N for level in pyramid] == [32, 16, 8]
    for parent, child in zip(pyramid, pyramid[1:]):
        keep = child.mr["idx"]
        eliminate = np.setdiff1d(np.arange(parent.N), keep)
        laplacian = parent.L.toarray()
        expected = laplacian[np.ix_(keep, keep)] - laplacian[
            np.ix_(keep, eliminate)
        ] @ np.linalg.solve(
            laplacian[np.ix_(eliminate, eliminate)], laplacian[np.ix_(eliminate, keep)]
        )
        np.testing.assert_allclose(child.L.toarray(), expected, atol=1e-10)
        np.testing.assert_array_equal(child.mr["orig_idx"], parent.mr["orig_idx"][keep])
        assert len(np.unique(child.mr["orig_idx"])) == child.N
        assert np.isfinite(child.W.data).all()
    coarse, original = utils.kron_coarsening(graph, r=0.5)
    assert coarse.N == 16 and original is graph
    signal = np.linspace(-1, 1, coarse.N)
    restored = utils.kron_interpolate(original, coarse, signal)
    assert restored.shape == (32,) and np.isfinite(restored).all()
    # PyGSP uses an approximate Chebyshev filter for regularized interpolation.
    np.testing.assert_allclose(restored[coarse.mr["idx"]], signal, atol=2e-3, rtol=0)


@pytest.mark.parametrize(
    "setting", [{"downsampling_method": "unknown"}, {"reduction_method": "unknown"}]
)
def test_pyramid_rejects_unsupported_algorithms(setting):
    with pytest.raises(NotImplementedError, match="Unknown graph"):
        utils.my_graph_multiresolution(graphs.Path(12), 1, sparsify=False, **setting)


def test_resistance_sparsification_agrees_for_graph_and_laplacian_and_keeps_psd():
    graph = graphs.Ring(32, k=2)
    np.random.seed(5)
    reduced = utils.graph_sparsify(graph, epsilon=0.5, maxiter=2)
    np.random.seed(5)
    laplacian = utils.graph_sparsify(graph.L, epsilon=0.5, maxiter=2).toarray()
    assert reduced.N == graph.N
    assert reduced.Ne <= graph.Ne
    np.testing.assert_allclose(laplacian, reduced.L.toarray(), atol=1e-10)
    np.testing.assert_allclose(laplacian, laplacian.T, atol=1e-12)
    np.testing.assert_allclose(laplacian.sum(1), 0, atol=1e-12)
    assert np.linalg.eigvalsh(laplacian).min() >= -1e-10
    assert np.all(reduced.W.toarray()[graph.W.toarray() == 0] == 0)


@pytest.mark.parametrize("epsilon", [0, 0.01, 1, 2])
def test_sparsification_rejects_invalid_approximation_ranges(epsilon):
    with pytest.raises(ValueError, match="Epsilon"):
        utils.graph_sparsify(graphs.Path(16), epsilon)


def test_sparsification_rejects_unsupported_laplacian():
    graph = graphs.Path(16)
    graph.compute_laplacian("normalized")
    with pytest.raises(NotImplementedError):
        utils.graph_sparsify(graph, 0.5)

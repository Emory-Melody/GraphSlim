"""Graph neural tangent kernel recursions: Gram matrices, intermediates, and options."""

import numpy as np
import pytest
import scipy.sparse as sp

from graphslim.models.gntk import GNTK


def small_graphs():
    rng = np.random.default_rng(0)
    path = sp.csr_matrix(np.eye(3) + np.eye(3, k=1) + np.eye(3, k=-1))
    star = sp.csr_matrix(
        np.array([[1.0, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], dtype=float)
    )
    pair = sp.csr_matrix(np.ones((2, 2)))
    return [(rng.random((a.shape[0], 3)), a) for a in (path, star, pair)]


@pytest.mark.parametrize("scale", ["uniform", "degree"])
@pytest.mark.parametrize("jk", [False, True])
def test_gntk_gram_over_graphs_is_symmetric_positive_semidefinite(scale, jk):
    kernel = GNTK(num_layers=2, num_mlp_layers=2, jk=jk, scale=scale)
    graphs = small_graphs()
    diags = [kernel.diag(x, a) for x, a in graphs]
    gram = np.array(
        [
            [kernel.gntk(x1, x2, d1, d2, a1, a2)[0] for (x2, a2), d2 in zip(graphs, diags)]
            for (x1, a1), d1 in zip(graphs, diags)
        ]
    )
    assert np.isfinite(gram).all()
    np.testing.assert_allclose(gram, gram.T, rtol=1e-10)
    assert np.linalg.eigvalsh(gram).min() > -1e-8 * np.abs(gram).max()


def test_gntk_returns_layerwise_intermediates_with_expected_shapes():
    kernel = GNTK(num_layers=3, num_mlp_layers=1, jk=False, scale="uniform")
    (x1, a1), (x2, a2), _ = small_graphs()
    value, sigmas, ntks, dots = kernel.gntk(
        x1, x2, kernel.diag(x1, a1), kernel.diag(x2, a2), a1, a2
    )
    assert len(sigmas) == 3 and len(ntks) == len(dots) == 2
    assert all(m.shape == (3, 4) for m in sigmas + ntks + dots)
    # Without jumping knowledge the value is twice the sum of the last tangent kernel.
    assert value == pytest.approx(2 * ntks[-1].sum())
    # ReLU derivative factors are arc-cosine probabilities in [0, 1].
    assert all(((d >= 0) & (d <= 1)).all() for d in dots)


def test_gntk_one_layer_kernel_is_aggregated_feature_inner_product():
    kernel = GNTK(num_layers=1, num_mlp_layers=1, jk=False, scale="uniform")
    (x, a), *_ = small_graphs()
    value, sigmas, ntks, dots = kernel.gntk(x, x, kernel.diag(x, a), kernel.diag(x, a), a, a)
    dense = a.toarray()
    np.testing.assert_allclose(sigmas[0], dense @ x @ x.T @ dense.T)
    assert value == pytest.approx(2 * (dense @ x @ x.T @ dense.T).sum())
    assert ntks == [] and dots == []


def test_gntk_diag_records_relu_standard_deviations():
    kernel = GNTK(num_layers=2, num_mlp_layers=1, jk=False, scale="uniform")
    (x, a), *_ = small_graphs()
    (diag,) = kernel.diag(x, a)
    dense = a.toarray()
    np.testing.assert_allclose(diag, np.sqrt(np.diag(dense @ x @ x.T @ dense.T)))


def test_gntk_rejects_unknown_scale():
    with pytest.raises(AssertionError):
        GNTK(num_layers=2, num_mlp_layers=1, jk=False, scale="sum")


def test_gntk_show_memory_lists_module_globals(capsys):
    GNTK(2, 1, False, "uniform").show_memory(unit="B", threshold=0)
    names = {line.split()[0] for line in capsys.readouterr().out.splitlines()}
    assert {"GNTK", "math", "np", "sp"} <= names
    GNTK(2, 1, False, "uniform").show_memory(unit="GB", threshold=1)
    assert capsys.readouterr().out == ""

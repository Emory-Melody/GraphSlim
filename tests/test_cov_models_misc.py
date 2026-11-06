"""Graphon generator, parameterized adjacency, random attacks, and PRBCD edge cases."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from deeprobust.graph.defense_pyg import GCN as SurrogateGCN
from torch_sparse import SparseTensor

from graphslim.models import IGNR, PGE, PRBCD, RandomAttack
from graphslim.models.ignr import EdgeBlock, Sine, get_mgrid, mx_inv, mx_inv_sqrt, mx_tr
from test_dataset_contracts import raw_graph


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


# ---------------------------------------------------------------- IGNR graphon


def test_inverse_matches_dense_csr_and_sparse_tensor_inputs():
    matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    expected = torch.linalg.inv(matrix)
    torch.testing.assert_close(mx_inv(matrix, "cpu"), expected)
    torch.testing.assert_close(mx_inv(sp.csr_matrix(matrix.numpy()), "cpu"), expected)
    torch.testing.assert_close(mx_inv(SparseTensor.from_dense(matrix), "cpu"), expected)


def test_inverses_suppress_the_smallest_near_zero_singular_value():
    matrix = torch.diag(torch.tensor([4.0, 1.0, 0.001]))
    torch.testing.assert_close(mx_inv(matrix, "cpu"), torch.diag(torch.tensor([0.25, 1.0, 0])))
    root, inverse = mx_inv_sqrt(matrix)
    torch.testing.assert_close(root, torch.diag(torch.tensor([0.5, 1.0, 0])))
    torch.testing.assert_close(inverse, torch.diag(torch.tensor([0.25, 1.0, 0])))
    root, inverse = mx_inv_sqrt(torch.diag(torch.tensor([4.0, 1.0])))
    torch.testing.assert_close(root @ root, inverse)
    assert mx_tr(matrix).item() == pytest.approx(5.001)


def test_coordinate_grids_span_minus_one_to_one():
    grid = get_mgrid(3)
    assert grid.shape == (9, 2) and grid.dtype == torch.float32
    torch.testing.assert_close(grid[0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(grid[-1], torch.tensor([1.0, 1.0]))
    cube = get_mgrid((1, 2, 3), dim=3)
    assert cube.shape == (6, 3)
    assert cube[:, 0].eq(-1).all() and cube[:, 2].unique().tolist() == [-1.0, 0.0, 1.0]
    with pytest.raises(NotImplementedError):
        get_mgrid(2, dim=4)


def test_activation_and_edge_block_shapes():
    x = torch.linspace(-1, 1, 5)
    torch.testing.assert_close(Sine()(x), torch.sin(30 * x))
    block = EdgeBlock(3, 4).eval()
    assert (block(torch.randn(6, 3)) >= 0).all()


def graphon_args():
    return SimpleNamespace(ep_ratio=0.5, sinkhorn_iter=2, mx_size=4)


def test_graphon_generates_symmetric_hollow_probabilities_and_differentiable_loss():
    torch.manual_seed(0)
    learner = IGNR(node_feature=3, nfeat=8, nnodes=5, device="cpu", args=graphon_args())
    features = torch.randn(5, 3)
    real = torch.tensor(
        [[2.0, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]]
    ) + torch.eye(4)
    adjacency, loss = learner(features, Lx=real)
    assert learner.training
    assert adjacency.shape == (5, 5)
    torch.testing.assert_close(adjacency, adjacency.T)
    assert adjacency.diagonal().eq(0).all()
    off_diagonal = adjacency[~torch.eye(5, dtype=torch.bool)]
    assert ((off_diagonal > 0) & (off_diagonal < 1)).all()
    torch.testing.assert_close(learner.Lx_inv, torch.linalg.inv(real))
    cached = learner.Lx_inv
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert torch.isfinite(learner.P.grad).all()
    learner(features, Lx=real * 2)
    assert learner.Lx_inv is cached
    inferred = learner.inference(features)
    assert not learner.training and inferred.shape == (5, 5)
    learner.reset_parameters()


# ------------------------------------------------------ parameterized adjacency


def pge_args(dataset, rate=0.5):
    return SimpleNamespace(dataset=dataset, reduction_rate=rate)


@pytest.mark.parametrize(
    "dataset,rate,width",
    [("cora", 0.5, 16), ("ogbn-arxiv", 0.5, 256), ("reddit", 0.001, 256), ("reddit", 0.01, 128)],
)
def test_pge_width_depends_on_dataset(dataset, rate, width):
    model = PGE(3, 4, nhid=16, nlayers=3, args=pge_args(dataset, rate))
    assert model.layers[0].out_features == width
    assert len(model.layers) == 3 and model.edge_index.shape == (2, 16)


def test_pge_outputs_symmetric_hollow_probabilities_and_chunked_path_matches():
    torch.manual_seed(1)
    features = torch.randn(5, 3)
    plain = PGE(3, 5, nhid=128, nlayers=3, args=pge_args("cora"))
    adjacency = plain(features)
    torch.testing.assert_close(adjacency, adjacency.T)
    assert adjacency.diagonal().eq(0).all()
    assert ((adjacency >= 0) & (adjacency < 1)).all()
    # With fixed batch-norm statistics, Reddit's five-way chunking is exact.
    chunked = PGE(3, 5, args=pge_args("reddit", 0.01))
    chunked.load_state_dict(plain.state_dict())
    plain.eval(), chunked.eval()
    torch.testing.assert_close(chunked.inference(features), plain.inference(features))


# ----------------------------------------------------------------- random attack


def symmetric_ring(n=8):
    rows = np.arange(n)
    adjacency = sp.coo_matrix((np.ones(n), (rows, (rows + 1) % n)), shape=(n, n))
    return (adjacency + adjacency.T).tocsr()


def edge_set(adjacency):
    return set(zip(*sp.triu(adjacency, k=1).nonzero()))


@pytest.mark.parametrize("kind", ["add", "remove", "flip"])
def test_random_structure_attack_changes_exactly_budgeted_symmetric_edges(kind):
    np.random.seed(3)
    adjacency = symmetric_ring()
    attacker = RandomAttack(nnodes=8)
    attacker.attack(adjacency, n_perturbations=3, type=kind.upper())
    modified = attacker.modified_adj.tocsr()
    assert abs(modified - modified.T).sum() == 0 and modified.diagonal().sum() == 0
    before, after = edge_set(adjacency), edge_set(modified)
    assert len(before ^ after) == 3
    if kind == "add":
        assert before < after
    elif kind == "remove":
        assert after < before
    assert attacker.modified_features is None


def test_random_feature_attack_zeroes_budgeted_columns_without_mutating_input():
    torch.manual_seed(4)
    features = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    attacker = RandomAttack(attack_structure=False, attack_features=True)
    attacker.attack(features, n_perturbations=2)
    zeroed = attacker.modified_features.eq(0).all(0)
    assert zeroed.sum() == 2 and features.ne(0).all()
    torch.testing.assert_close(attacker.modified_features[:, ~zeroed], features[:, ~zeroed])
    assert attacker.modified_adj is None


def test_random_attack_rejects_unknown_types_and_node_injection():
    attacker = RandomAttack()
    with pytest.raises(AssertionError):
        attacker.perturb_adj(symmetric_ring(), 1, type="rewire")
    with pytest.raises(NotImplementedError):
        attacker.inject_nodes(symmetric_ring(), 1, 1)


# ------------------------------------------------------------------------ PRBCD


def prbcd(**kwargs):
    options = dict(
        device="cpu",
        search_space_size=60,
        max_final_samples=3,
        epochs=3,
        fine_tune_epochs=1,
        pretrain_epochs=2,
    )
    options.update(kwargs)
    return PRBCD(raw_graph(), **options)


def test_default_surrogate_is_pretrained_without_reading_test_labels(capsys):
    torch.manual_seed(5)
    attack = prbcd()
    assert isinstance(attack.model, SurrogateGCN) and not attack.model.training
    assert attack.model.data is not None
    assert attack.n == 20 and attack.n_possible_edges == 190


def test_attack_on_supplied_edges_stays_within_flip_budget():
    torch.manual_seed(6)
    data = raw_graph()
    model = SurrogateGCN(nfeat=3, nhid=4, nclass=2, dropout=0, device="cpu")
    attack = PRBCD(
        data,
        model=model,
        device="cpu",
        search_space_size=60,
        max_final_samples=3,
        epochs=2,
        fine_tune_epochs=1,
        pretrain_epochs=2,
    )
    attack.pretrain_model(model)
    half = data.edge_index[:, ::2]
    edges, weights = attack.attack(edge_index=half, ptb_rate=0.25)
    assert torch.equal(attack.edge_index, half)
    clean = {tuple(e) for e in half.t().tolist()}
    changed = clean ^ {tuple(e) for e in edges.t().tolist()}
    budget = int(0.25 * half.shape[1] // 2)
    assert len(changed) <= 2 * budget and weights.eq(1).all()


def test_final_sampling_skips_draws_above_the_budget(capsys):
    torch.manual_seed(7)
    model = SurrogateGCN(nfeat=3, nhid=4, nclass=2, dropout=0, device="cpu")
    attack = prbcd(model=model)
    attack.edge_index = attack.data.edge_index
    attack.edge_weight = torch.ones(attack.edge_index.shape[1])
    attack.sample_random_block(2)
    attack.perturbed_edge_weight = torch.ones_like(attack.perturbed_edge_weight)
    edges, weights = attack.sample_final_edges(2)
    assert "too many samples" in capsys.readouterr().out
    assert attack.perturbed_edge_weight.sum() == 2
    assert weights.eq(1).all()


@pytest.mark.parametrize("undirected", [True, False])
def test_resampling_keeps_heavy_candidates_or_reports_an_impossible_budget(undirected):
    torch.manual_seed(8)
    model = SurrogateGCN(nfeat=3, nhid=4, nclass=2, dropout=0, device="cpu")
    attack = prbcd(model=model, make_undirected=undirected, search_space_size=30)
    attack.sample_random_block(2)
    weights = torch.linspace(0, 1, attack.perturbed_edge_weight.numel())
    attack.perturbed_edge_weight = weights.clone()
    heavy = attack.current_search_space[weights > 0.5].tolist()
    attack.resample_random_block(2)
    assert set(heavy) <= set(attack.current_search_space.tolist())
    kept = dict(zip(attack.current_search_space.tolist(), attack.perturbed_edge_weight.tolist()))
    assert all(kept[i] > 0.5 for i in heavy)
    with pytest.raises(RuntimeError, match="Sampling random block"):
        attack.resample_random_block(10_000)


def test_scheduled_direct_update_adds_scaled_gradient_and_clamps():
    model = SurrogateGCN(nfeat=3, nhid=4, nclass=2, dropout=0, device="cpu")
    attack = prbcd(model=model, epochs=4, fine_tune_epochs=2)
    attack.lr_factor = 4.0
    attack.perturbed_edge_weight = torch.tensor([0.5, 0.5])
    attack._update_edge_weights(5, 3, torch.tensor([1.0, -10.0]))
    lr = float(5 / 20 / 2 * 4.0 / np.sqrt(2))
    torch.testing.assert_close(attack.perturbed_edge_weight, torch.tensor([0.5 + lr, attack.eps]))

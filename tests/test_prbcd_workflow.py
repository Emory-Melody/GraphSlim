"""Real short adversarial optimization plus budget and gradient contracts."""

import pytest
import torch
from deeprobust.graph.defense_pyg import GCN

from graphslim.models.prbcd import (
    PRBCD,
    entropy,
    grad_with_checkpoint,
    linear_to_full_idx,
    linear_to_triu_idx,
    softmax_entropy,
    to_symmetric,
)
from test_dataset_contracts import raw_graph


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def agent(undirected=True, **kwargs):
    data = raw_graph()
    model = GCN(nfeat=3, nhid=4, nclass=2, dropout=0, device="cpu")
    return PRBCD(
        data,
        model=model,
        make_undirected=undirected,
        device="cpu",
        search_space_size=60,
        max_final_samples=4,
        epochs=3,
        fine_tune_epochs=1,
        pretrain_epochs=3,
        **kwargs,
    )


@pytest.mark.parametrize("undirected", [False, True])
def test_actual_surrogate_training_attack_and_binary_graph_budget(undirected):
    torch.manual_seed(12)
    attack = agent(undirected)
    attack.pretrain_model(attack.model)
    original = attack.data.edge_index.clone()
    original_labels = attack.data.y.clone()
    edges, weights = attack.attack(ptb_rate=0.2)
    assert weights.eq(1).all()
    assert edges.shape[1] == weights.numel()
    assert not (edges[0] == edges[1]).any()
    dense = torch.zeros(20, 20)
    dense[edges[0], edges[1]] = weights
    clean = torch.zeros_like(dense)
    clean[original[0], original[1]] = 1
    assert (dense != clean).sum() <= 8
    if undirected:
        torch.testing.assert_close(dense, dense.T)
    assert torch.equal(attack.data.y, original_labels)
    assert torch.equal(attack.data.edge_index, original)
    output = attack.model.predict(attack.data.x, edges, weights)
    loss, score = attack.get_perf(output, attack.data.y, attack.data.val_mask)
    assert loss >= 0 and 0 <= score <= 1


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("budget", [1, 10])
def test_projection_preserves_input_when_requested_and_enforces_edge_budget(inplace, budget):
    attack = agent()
    values = torch.tensor([-0.2, 0.3, 0.8, 1.4])
    original = values.clone()
    projected = attack.project(budget, values, 1e-7, inplace=inplace)
    assert projected.min() >= 0 and projected.max() <= 1
    assert projected.sum() <= budget + 1e-5
    assert (projected.data_ptr() == values.data_ptr()) == inplace
    if not inplace:
        assert torch.equal(values, original)


def test_impossible_random_block_budget_reports_error():
    attack = agent()
    attack.search_space_size = 2
    with pytest.raises(RuntimeError, match="Sampling random block"):
        attack.sample_random_block(3)


def test_loss_variants_and_entropy_have_finite_gradients():
    attack = agent()
    logits = torch.tensor([[2.0, -1], [-1, 2]], requires_grad=True)
    labels = torch.tensor([0, 1])
    for kind in ["CE", "MCE", "NCE", "tanhMargin"]:
        loss = attack.loss_attack(logits, labels, kind)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
        assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(softmax_entropy(logits), entropy(logits.log_softmax(1)))
    torch.testing.assert_close(
        softmax_entropy(torch.zeros(2, 2)), torch.full((2,), torch.log(torch.tensor(2.0)))
    )


def test_index_mappings_cover_all_directed_and_undirected_pairs():
    for n in [2, 3, 8]:
        actual = linear_to_triu_idx(n, torch.arange(n * (n - 1) // 2))
        assert torch.equal(actual, torch.triu_indices(n, n, offset=1))
        full = linear_to_full_idx(n, torch.arange(n * n))
        assert len(set(zip(full[0].tolist(), full[1].tolist()))) == n * n
    edges = torch.tensor([[0, 1], [1, 2]])
    result, weights = to_symmetric(edges, torch.tensor([0.3, 0.8]), 3)
    dense = torch.zeros(3, 3)
    dense[result[0], result[1]] = weights
    torch.testing.assert_close(dense, dense.T)


def test_checkpoint_gradient_handles_nonleaf_inputs_without_accumulation():
    x = torch.tensor([1.0, 2], requires_grad=True)
    transformed = 2 * x
    gradient = grad_with_checkpoint(transformed.square().sum(), [transformed])[0]
    torch.testing.assert_close(gradient, 2 * transformed)
    torch.testing.assert_close(transformed.grad, torch.zeros(2))

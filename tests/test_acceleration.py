from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from graphslim.acceleration import CUDAGraphStep, acceleration_records, method_acceleration_support
from graphslim.reduction import list_methods


def options(enabled=True):
    return SimpleNamespace(cuda_graph=enabled, cuda_graph_min_steps=1, cuda_graph_warmup=3)


def test_all_registered_methods_have_stage_policy():
    rows = method_acceleration_support()
    assert {r["method"] for r in rows} == set(list_methods())
    assert all(r["downstream"] == "shared_first_order_training" for r in rows)
    assert next(r for r in rows if r["method"] == "gcond")["reduction"] == "eager_second_order"
    assert (
        next(r for r in rows if r["method"] == "nope")["reduction"] == "eager_cpu_or_noniterative"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_never_uses_cuda(monkeypatch, enabled):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *a: pytest.fail("CPU touched CUDA"))
    args = options(enabled)
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    step = CUDAGraphStep(lambda: parameter.square().sum(), [parameter], args, "test", 200)
    for _ in range(3):
        optimizer.zero_grad()
        step()
        optimizer.step()
    assert torch.allclose(parameter, torch.full((3,), 0.8**3))
    assert step.record["reason"] == ("cpu" if enabled else "not_requested")
    assert len(acceleration_records(args)) == int(enabled)


def test_cuda_graph_cli_defaults_and_cpu_flag(tmp_path):
    from graphslim.config import get_args

    base = ["-G", "-1", "--save_path", str(tmp_path)]
    assert not get_args(base).cuda_graph
    args = get_args(base + ["--cuda_graph", "--cuda_graph_min_steps", "10"])
    assert args.cuda_graph and args.cuda_graph_min_steps == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batchnorm,dropout", [(False, 0.0), (True, 0.0), (False, 0.3)])
def test_gpu_training_preserves_updates_buffers_and_rng(batchnorm, dropout):
    device = "cuda:0"
    torch.manual_seed(31)
    layers = [torch.nn.Linear(8, 16)]
    if batchnorm:
        layers.append(torch.nn.BatchNorm1d(16))
    layers.extend([torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(16, 2)])
    original = torch.nn.Sequential(*layers).to(device)
    features = torch.randn(24, 8, device=device)
    targets = torch.arange(24, device=device) % 2

    def fit(enabled):
        model = deepcopy(original).train()
        torch.manual_seed(19)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        args = options(enabled)

        def objective():
            output = model(features)
            return output, torch.nn.functional.cross_entropy(output, targets)

        step = CUDAGraphStep(
            objective,
            model.parameters(),
            args,
            "test",
            20,
            modules=(model,),
            inputs=(features, targets),
        )
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            _, loss = step()
            losses.append(loss.item())
            optimizer.step()
        return model.state_dict(), losses, torch.cuda.get_rng_state(), step.record

    eager, captured = fit(False), fit(True)
    assert eager[1] == captured[1]
    assert torch.equal(eager[2], captured[2])
    assert all(torch.equal(eager[0][key], captured[0][key]) for key in eager[0])
    if not batchnorm and not dropout:
        assert captured[3]["status"] == "captured", str(captured[3])
        assert captured[3]["replays"] == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_failure_restores_reference_update():
    parameter = torch.nn.Parameter(torch.ones(4, device="cuda:0"))
    args = options()

    def objective():
        loss = parameter.square().sum()
        loss.item()  # Forbidden CPU synchronization during capture.
        return loss

    step = CUDAGraphStep(objective, [parameter], args, "unsafe", 100)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        step()
        optimizer.step()
    assert torch.equal(parameter, torch.full_like(parameter, 0.64))
    assert step.record["status"] == "eager"
    assert step.record["reason"].startswith(("capture_rejected", "unsafe_operations"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_short_stage_does_not_capture():
    args = options()
    args.cuda_graph_min_steps = 64
    parameter = torch.nn.Parameter(torch.ones(3, device="cuda:0"))
    step = CUDAGraphStep(lambda: parameter.sum(), [parameter], args, "short", 2)
    step()
    assert step.record["reason"] == "too_few_steps"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_released_sgnk_matches_author_arithmetic(dtype):
    from pathlib import Path
    from graphslim.condensation.gcgp import released_neural_kernel

    author_path = Path("runs/reference/GCGP/gcgp/sgnk.py")
    if not author_path.exists():
        pytest.skip("Optional pinned reference checkout unavailable")
    from benchmark.verify_lightweight import load_definitions

    author = load_definitions(author_path, {"SimplifyingGraphNeuralKernel"})[
        "SimplifyingGraphNeuralKernel"
    ](L=2)
    generator = torch.Generator().manual_seed(7)
    query = torch.rand(17, 11, generator=generator).to(dtype)
    synthetic = torch.rand(7, 11, generator=generator).to(dtype).requires_grad_(True)
    left = author.nodes_gram(query, synthetic, None, None)
    right = released_neural_kernel(query, synthetic)
    assert torch.equal(left, right)
    a = torch.autograd.grad(left.square().mean(), synthetic)[0]
    b = torch.autograd.grad(right.square().mean(), synthetic)[0]
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_partial_kernel_capture_preserves_long_gp_trajectory(dtype):
    from graphslim.acceleration import CUDAGraphCallable
    from graphslim.condensation.gcgp import released_neural_kernel, gp_predict

    generator = torch.Generator().manual_seed(3)
    query = torch.rand(24, 16, generator=generator).to(device="cuda:0", dtype=dtype)
    initial = torch.rand(8, 16, generator=generator).to(device="cuda:0", dtype=dtype)
    initial_targets = torch.rand(8, 2, generator=generator).to(device="cuda:0", dtype=dtype)
    target = torch.nn.functional.one_hot(torch.arange(24, device="cuda:0") % 2, 2).to(dtype)

    def fit(enabled):
        feature = torch.nn.Parameter(initial.clone())
        labels = torch.nn.Parameter(initial_targets.clone())
        optimizer = torch.optim.Adam([feature, labels], lr=0.01)
        kernels = CUDAGraphCallable(
            lambda x, y: (released_neural_kernel(y, y), released_neural_kernel(x, y)),
            options(enabled),
            "test.kernels",
            100,
        )
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            ss, ts = kernels(query, feature)
            loss = torch.nn.functional.mse_loss(gp_predict(ts, ss, labels, 0.5, True), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return feature.detach(), labels.detach(), losses, kernels.record

    eager, captured = fit(False), fit(True)
    assert eager[2] == captured[2]
    assert torch.equal(eager[0], captured[0]) and torch.equal(eager[1], captured[1])
    assert captured[3]["status"] == "captured", captured[3]

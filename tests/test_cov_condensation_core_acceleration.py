"""CPU-checkable parts of the optional CUDA-graph helpers: gating, replay bookkeeping, fallbacks.

Real capture needs a GPU (see tests/test_acceleration.py). Here a parameter subclass that
*reports* a CUDA device drives the pure-Python eligibility checks, and an eager stand-in for
the graphed callable drives the replay/invalidation bookkeeping.
"""

from types import SimpleNamespace

import pytest
import torch

from graphslim.acceleration import (
    CUDAGraphCallable,
    CUDAGraphStep,
    _LossModule,
    acceleration_records,
)


class CudaReportingParameter(torch.nn.Parameter):
    @property
    def device(self):
        return torch.device("cuda", 0)


class CudaReportingTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda", 0)


def cuda_parameter(value=None):
    return CudaReportingParameter(torch.ones(2) if value is None else value)


def options(**extra):
    return SimpleNamespace(cuda_graph=True, cuda_graph_min_steps=4, cuda_graph_warmup=2, **extra)


def reason(parameters, steps=10, **kwargs):
    step = CUDAGraphStep(lambda: None, parameters, options(), "stage", steps, **kwargs)
    return step.record["status"], step.record["reason"]


def test_eligibility_checks_report_the_first_blocking_reason():
    good = cuda_parameter()
    assert reason([good], blocked_reason="second_order") == ("eager", "second_order")
    assert reason([good, torch.nn.Parameter(torch.ones(2))]) == (
        "eager",
        "mixed_or_missing_parameters",
    )
    assert reason([good], steps=3) == ("eager", "too_few_steps")
    sparse = CudaReportingParameter(torch.eye(2).to_sparse())
    assert reason([sparse]) == ("eager", "non_leaf_or_sparse_parameters")
    plain = torch.Tensor._make_subclass(CudaReportingTensor, torch.ones(2), True)
    assert reason([plain]) == ("eager", "unregistered_parameter_tensor")
    loose = torch.ones(2, requires_grad=True)
    assert reason([good], inputs=(loose, 3)) == ("eager", "untracked_input_gradient")
    module = torch.nn.Linear(2, 2)
    assert reason([good], modules=(module,)) == ("eager", "untracked_module_parameters")
    assert reason([good], inputs=(torch.ones(2), good)) == ("pending", None)
    # Frozen parameters are dropped; nothing trainable means CPU fallback.
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    step = CUDAGraphStep(lambda: None, [frozen], options(), "stage", 10)
    assert step.parameters == () and step.record["reason"] == "cpu"


def test_records_are_shared_only_when_requested_or_traced():
    parameter = torch.nn.Parameter(torch.ones(1))
    quiet = SimpleNamespace(cuda_graph=False)
    CUDAGraphStep(lambda: parameter.sum(), [parameter], quiet, "a", 10)
    assert acceleration_records(quiet) == []
    traced = SimpleNamespace(cuda_graph=False, _cuda_graph_trace=True)
    step = CUDAGraphStep(lambda: (parameter * 2, parameter.sum()), [parameter], traced, "b", 10)
    output, loss = step()
    step()
    assert acceleration_records(traced)[0]["loss_trace"] == [1.0, 1.0]
    assert step.record["eager_steps"] == 2 and parameter.grad.item() == 2.0
    torch.testing.assert_close(output, parameter.detach() * 2)


def test_loss_module_registers_parameters_and_returns_scalar_loss():
    weight = torch.nn.Parameter(torch.tensor([2.0]))
    single = _LossModule(lambda: weight.square().sum(), [weight])
    assert list(single.parameters()) == [weight]
    assert single().item() == 4.0
    paired = _LossModule(lambda: (weight * 3, weight.sum()), [weight])
    assert paired().item() == 2.0 and paired.last_output[0].item() == 6.0


def eager_replay_step(module, with_tuple):
    parameter = next(module.parameters())
    features = torch.ones(3, 2)

    def objective():
        output = module(features)
        loss = output.square().mean()
        return (output, loss) if with_tuple else loss

    args = SimpleNamespace(cuda_graph=False)
    step = CUDAGraphStep(objective, module.parameters(), args, "replay", 10, modules=(module,))
    wrapper = _LossModule(objective, step.parameters)
    # The eager wrapper stands in for the graphed callable produced on a GPU.
    step.wrapper, step.graph = wrapper, wrapper
    step.signature = step._signature()
    return step, parameter, objective


@pytest.mark.parametrize("with_tuple", [False, True])
def test_replay_bookkeeping_and_invalidation_on_mode_change(with_tuple):
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    step, parameter, objective = eager_replay_step(module, with_tuple)
    result = step()
    expected = objective()
    loss = result[-1] if with_tuple else result
    torch.testing.assert_close(loss, expected[-1] if with_tuple else expected)
    if with_tuple:
        assert not result[0].requires_grad
    assert step.record["replays"] == 1 and step.gradients[0] is parameter.grad
    assert parameter.grad.abs().sum() > 0
    captures = []
    step._capture = lambda: captures.append(True) or step._execute()
    module.eval()  # mode change alters the signature: the graph must be dropped
    step()
    assert step.graph is None and step.record["invalidations"] == 1
    assert step.record["reason"] == "parameter_storage_or_mode_changed" and captures == [True]


def test_buffers_are_deduplicated_across_modules():
    bn = torch.nn.BatchNorm1d(2)
    parameter = torch.nn.Parameter(torch.ones(1))
    step = CUDAGraphStep(
        lambda: parameter.sum(), [parameter], SimpleNamespace(), "s", 1, modules=(bn, bn)
    )
    assert len(step._buffers()) == 3
    tensors, modes = step._signature()
    assert len(tensors) == 4 and modes[0][1] is True


def test_kernel_callable_falls_back_to_eager_on_cpu():
    calls = []

    def kernel(x, y):
        calls.append(1)
        return x @ y.T

    x, y = torch.rand(3, 2, requires_grad=True), torch.rand(4, 2)
    quiet = SimpleNamespace(cuda_graph=False)
    off = CUDAGraphCallable(kernel, quiet, "k", 100)
    assert off.record["reason"] == "not_requested"
    torch.testing.assert_close(off(x, y), x @ y.T)
    assert off.record["status"] == "eager" and acceleration_records(quiet) == []
    short = CUDAGraphCallable(kernel, options(), "k", 1)
    assert short.record["reason"] == "too_few_steps"
    args = options()
    requested = CUDAGraphCallable(kernel, args, "k", 100)
    assert requested.record["status"] == "pending"
    torch.testing.assert_close(requested(x, y), x @ y.T)
    assert requested.record == acceleration_records(args)[0]
    assert requested.record["status"] == "eager"
    assert requested.record["reason"] == "non_dense_or_cpu_input"
    assert len(calls) == 2


# --------------------------------------------------------------------------- capture logic
# The CUDA runtime calls are replaced by eager stand-ins so the acceptance/rejection
# logic around capture (exact comparison, state restore, fallbacks) runs on CPU.


class FakeStream:
    def wait_stream(self, other):
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    from contextlib import nullcontext

    state = {"graphed": lambda function, *a, **k: function}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: FakeStream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *a: torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *a: None)
    monkeypatch.setattr(
        torch.cuda, "make_graphed_callables", lambda *a, **k: state["graphed"](*a, **k)
    )
    return state


def captured_step(objective, parameter):
    return CUDAGraphStep(objective, [parameter], options(), "capture", 10)


def test_capture_is_accepted_only_after_identical_replay(fake_cuda):
    parameter = cuda_parameter(torch.tensor([1.0, 2.0]))
    step = captured_step(lambda: (parameter * 2, parameter.square().sum()), parameter)
    output, loss = step()
    assert step.record["status"] == "captured" and loss.item() == 5.0
    torch.testing.assert_close(parameter.grad, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(output, torch.tensor([2.0, 4.0]))
    step()
    assert step.record["replays"] == 2 and step.record["capture_seconds"] >= 0


def test_capture_rejects_host_synchronizing_operations(fake_cuda):
    parameter = cuda_parameter()
    step = captured_step(lambda: parameter.sum() * float(parameter.sum().item()), parameter)
    loss = step()
    assert loss.item() == 4.0
    assert step.record["status"] == "eager"
    assert (
        step.record["reason"].startswith("unsafe_operations")
        and "aten::item" in step.record["reason"]
    )
    step()
    assert step.record["eager_steps"] == 2


@pytest.mark.parametrize("failure", ["error", "mismatch"])
def test_rejected_capture_restores_reference_state(fake_cuda, failure):
    parameter = cuda_parameter(torch.tensor([1.0, 3.0]))

    def broken(*args, **kwargs):
        if failure == "error":
            raise RuntimeError("capture unsupported\nmore detail")

        def wrong():
            return parameter.sum() * 0

        return wrong

    fake_cuda["graphed"] = broken
    step = captured_step(lambda: parameter.square().sum(), parameter)
    loss = step()
    assert loss.item() == 10.0 and step.graph is None
    torch.testing.assert_close(parameter.grad, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(parameter.detach(), torch.tensor([1.0, 3.0]))
    expected = "capture unsupported" if failure == "error" else "nonidentical"
    assert step.record["status"] == "eager" and expected in step.record["reason"]
    assert step.record["errors"]


def cuda_input(value, grad=False):
    return torch.Tensor._make_subclass(CudaReportingTensor, value, grad)


def test_kernel_capture_replays_and_invalidates_on_shape_change(fake_cuda):
    calls = []

    def kernel(x, y):
        calls.append(x.shape)
        return x @ y.T, (x * x).sum()

    callable_ = CUDAGraphCallable(kernel, options(), "k", 100)
    x, y = cuda_input(torch.rand(3, 2), True), cuda_input(torch.rand(4, 2))
    product, norm = callable_(x, y)
    assert callable_.record["status"] == "captured"
    torch.testing.assert_close(
        product.as_subclass(torch.Tensor), (x @ y.T).as_subclass(torch.Tensor)
    )
    norm.backward()
    assert x.grad is not None
    callable_(x, y)
    assert callable_.record["replays"] == 2
    wider = cuda_input(torch.rand(5, 2), True)
    callable_(wider, y)
    assert callable_.record["invalidations"] == 1
    frozen = CUDAGraphCallable(kernel, options(), "k", 100)
    frozen(cuda_input(torch.rand(2, 2)), y)  # nothing differentiable: plain eager call
    assert frozen.graphed is None and frozen.record["status"] == "pending"


@pytest.mark.parametrize("failure", ["error", "mismatch"])
def test_kernel_capture_falls_back_to_reference(fake_cuda, failure):
    def kernel(x):
        return x * 2

    def broken(function, inputs, **kwargs):
        if failure == "error":
            raise ValueError("bad kernel")
        return lambda x: x * 3

    fake_cuda["graphed"] = broken
    callable_ = CUDAGraphCallable(kernel, options(), "k", 100)
    x = cuda_input(torch.ones(2), True)
    result = callable_(x)
    torch.testing.assert_close(result.as_subclass(torch.Tensor), torch.full((2,), 2.0))
    expected = "bad kernel" if failure == "error" else "nonidentical_kernel_or_gradient"
    assert callable_.graphed is None and expected in callable_.record["reason"]

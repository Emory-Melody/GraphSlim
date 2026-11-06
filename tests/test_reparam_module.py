"""Parameter flattening must preserve predictions, gradients, and temporary state."""

from copy import deepcopy

import pytest
import torch

from graphslim.models.reparam_module import ReparamModule


def test_explicit_parameters_preserve_predictions_and_gradients():
    torch.manual_seed(8)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh(), torch.nn.Linear(6, 2))
    reference = deepcopy(model)
    wrapped = ReparamModule(model)
    assert [name for name, _ in wrapped.named_parameters()] == ["flat_param"]
    assert wrapped.param_numel == 4 * 6 + 6 + 6 * 2 + 2
    expected_flat = torch.cat([p.detach().reshape(-1) for p in reference.parameters()])
    torch.testing.assert_close(wrapped.flat_param.detach(), expected_flat)
    x = torch.randn(3, 4)
    expected = reference(x)
    replacement = wrapped.flat_param.detach().clone().requires_grad_()
    actual = wrapped(x, flat_param=replacement)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    gradients = torch.cat([p.grad.flatten() for p in reference.parameters()])
    torch.testing.assert_close(replacement.grad, gradients)
    # A row-vector parameter is squeezed to the flat layout.
    torch.testing.assert_close(wrapped(x, flat_param=replacement[None, :]), expected)


def test_replacement_parameters_change_the_function():
    wrapped = ReparamModule(torch.nn.Linear(2, 1))
    x = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    # flat layout is [w00, w01, b]
    out = wrapped(x, flat_param=torch.tensor([2.0, -1.0, 0.5]))
    torch.testing.assert_close(out, torch.tensor([[0.5], [7.5]]))


def test_tied_weights_are_preserved_under_replacement():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(3, 3, bias=False)
            self.second = torch.nn.Linear(3, 3, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x):
            return self.second(self.first(x))

    model = Tied()
    wrapped = ReparamModule(model)
    assert wrapped.param_numel == 9
    assert wrapped._shared_param_infos == (("module.second", "weight", "module.first", "weight"),)
    identity = torch.eye(3).flatten()
    x = torch.randn(2, 3)
    torch.testing.assert_close(wrapped(x, flat_param=identity), x)
    torch.testing.assert_close(wrapped(x, flat_param=2 * identity), 4 * x)
    assert model.first.weight is model.second.weight


def test_temporary_parameters_and_buffers_restore_on_normal_exit():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).eval()
    wrapped = ReparamModule(model)
    assert [(mn, n) for mn, n, _ in wrapped._buffer_infos] == [
        ("module.1", "running_mean"),
        ("module.1", "running_var"),
        ("module.1", "num_batches_tracked"),
    ]
    weight = model[0].weight
    mean = model[1].running_mean
    zeros = torch.zeros_like(wrapped.flat_param)
    with wrapped.unflattened_param(zeros):
        with wrapped.replaced_buffers([torch.ones_like(b) for _, _, b in wrapped._buffer_infos]):
            assert model[0].weight is not weight
            assert torch.equal(model[0].weight, torch.zeros(3, 3))
            assert torch.equal(model[1].running_mean, torch.ones(3))
    assert model[0].weight is weight
    assert model[1].running_mean is mean


def test_buffers_replace_batchnorm_statistics_in_forward():
    torch.manual_seed(2)
    wrapped = ReparamModule(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2)))
    wrapped.eval()
    x = torch.randn(4, 3)
    parameter = wrapped.flat_param.detach().clone()
    base = wrapped(x, flat_param=parameter)
    shifted = [b.clone() for _, _, b in wrapped._buffer_infos]
    shifted[0] = shifted[0] + 1.0  # running mean; running var stays 1
    out = wrapped(x, flat_param=parameter, buffers=shifted)
    # BatchNorm weight=1, bias=0 at init: shifting the mean shifts outputs by -1/sqrt(1+eps).
    torch.testing.assert_close(base - out, torch.full_like(out, 1 / (1 + 1e-5) ** 0.5))


def test_trace_matches_eager_for_parameters_and_buffers():
    torch.manual_seed(4)
    wrapped = ReparamModule(
        torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2)).eval()
    )
    x = torch.randn(4, 3)
    parameter = wrapped.flat_param.detach().clone()
    buffers = [b.clone() for _, _, b in wrapped._buffer_infos]
    eager = wrapped(x, flat_param=parameter, buffers=buffers)
    assert wrapped.trace(x) is wrapped
    torch.testing.assert_close(wrapped(x, flat_param=parameter, buffers=buffers), eager)
    torch.testing.assert_close(wrapped(x, flat_param=parameter), eager)
    assert wrapped.to("cpu") is wrapped
    assert wrapped.double() is wrapped  # converts the traced copy
    with pytest.raises(AssertionError, match="already traced"):
        wrapped.trace(x)


def test_clear_views_and_module_lookup():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    wrapped = ReparamModule(model)
    assert wrapped._get_module_from_name("") is wrapped
    assert wrapped._get_module_from_name("module.0") is model[0]
    assert isinstance(model[0].weight, torch.Tensor)
    wrapped.clear_views()
    assert model[0].weight is None and model[0].bias is None


def test_mixed_parameter_dtypes_are_rejected():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2).double())
    with pytest.raises(AssertionError, match="same dtype"):
        ReparamModule(model)


def test_untraced_dtype_conversion_and_tuple_trace_input():
    torch.manual_seed(1)
    wrapped = ReparamModule(torch.nn.Linear(3, 2))
    reference = wrapped.flat_param.detach().clone()
    assert wrapped.double() is wrapped
    assert wrapped.flat_param.dtype == torch.float64
    torch.testing.assert_close(wrapped.flat_param.detach().float(), reference)
    x = torch.randn(4, 3, dtype=torch.float64)
    eager = wrapped(x, flat_param=wrapped.flat_param)
    assert eager.dtype == torch.float64
    wrapped.trace((x,))  # positional inputs may be given as a tuple
    torch.testing.assert_close(wrapped(x, flat_param=wrapped.flat_param), eager)


def test_default_forward_uses_own_flat_parameter():
    # Regression: forward() squeezed ``flat_param`` before checking for None.
    torch.manual_seed(8)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh(), torch.nn.Linear(6, 2))
    reference = deepcopy(model)
    wrapped = ReparamModule(model)
    x = torch.randn(3, 4)
    out = wrapped(x)
    torch.testing.assert_close(out, reference(x))
    out.sum().backward()
    assert wrapped.flat_param.grad is not None
    assert wrapped.flat_param.grad.shape == (wrapped.param_numel,)


def test_temporary_parameters_and_buffers_restore_after_exception():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).eval()
    wrapped = ReparamModule(model)
    weight = model[0].weight
    mean = model[1].running_mean
    with pytest.raises(RuntimeError, match="abort"):
        with wrapped.unflattened_param(torch.zeros_like(wrapped.flat_param)):
            with wrapped.replaced_buffers(
                [torch.ones_like(b) for _, _, b in wrapped._buffer_infos]
            ):
                assert model[0].weight is not weight
                raise RuntimeError("abort")
    assert model[0].weight is weight
    assert model[1].running_mean is mean


def test_single_parameter_replacement_remains_a_flat_vector():
    wrapped = ReparamModule(torch.nn.Linear(1, 1, bias=False))
    x = torch.tensor([[2.0], [4.0]])
    torch.testing.assert_close(wrapped(x, flat_param=torch.tensor([3.0])), 3 * x)
    torch.testing.assert_close(wrapped(x, flat_param=torch.tensor([[3.0]])), 3 * x)

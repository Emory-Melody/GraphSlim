"""Optional native CUDA graphs for first-order, fixed-input training stages.

Optimizers, validation, sampling and higher-order differentiation stay eager.
Capture is accepted only after an exact forward/backward/state/RNG comparison.
Warmup must not add updates, advance randomness or alter BatchNorm statistics.
"""

from __future__ import annotations

import time
from typing import Callable

import torch


def acceleration_records(args):
    """JSON-compatible stage records; a requested backend is not a speed claim."""
    return list(getattr(args, "_cuda_graph_records", ()))


class _LossModule(torch.nn.Module):
    """Expose closure parameters to native make_graphed_callables."""

    def __init__(self, function, parameters):
        super().__init__()
        self.function = function
        self.last_output = None
        for index, parameter in enumerate(parameters):
            self.register_parameter(f"parameter_{index}", parameter)

    def forward(self):
        self.last_output = self.function()
        # Capture only the scalar loss; retain logits/diagnostics for reads.
        return self.last_output[-1] if isinstance(self.last_output, tuple) else self.last_output


class CUDAGraphStep:
    """A fixed-input loss (or flat tensor tuple ending in loss) and backward.

    Call after ``optimizer.zero_grad()`` and before its normal ``step()``.
    The callable must use only Torch operations and have no Python side effects.
    Explicit parameters and modules own all mutable tensor state. Create a new
    instance when inputs/control flow change. First-order gradients only.
    """

    def __init__(
        self,
        function: Callable,
        parameters,
        args,
        stage,
        steps,
        modules=(),
        inputs=(),
        blocked_reason=None,
    ):
        self.function = function
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        self.modules = tuple(modules)
        self.inputs = tuple(inputs)
        self.warmups = getattr(args, "cuda_graph_warmup", 3)
        self.trace = bool(getattr(args, "_cuda_graph_trace", False))
        self.graph = None
        self.wrapper = None
        self.outputs = None
        self.gradients = ()
        self.signature = None
        self.record = {
            "stage": stage,
            "requested": bool(getattr(args, "cuda_graph", False)),
            "status": "eager",
            "reason": "not_requested",
            "expected_steps": steps,
            "capture_seconds": 0.0,
            "verification_seconds": 0.0,
            "replays": 0,
            "eager_steps": 0,
            "invalidations": 0,
        }
        self.device = self.parameters[0].device if self.parameters else torch.device("cpu")
        if self.record["requested"] or self.trace:
            if not hasattr(args, "_cuda_graph_records"):
                args._cuda_graph_records = []
            args._cuda_graph_records.append(self.record)
        if self.record["requested"]:
            if self.device.type != "cuda":
                self.record["reason"] = "cpu"
            elif blocked_reason:
                self.record["reason"] = blocked_reason
            elif not self.parameters or any(p.device != self.device for p in self.parameters):
                self.record["reason"] = "mixed_or_missing_parameters"
            elif steps < getattr(args, "cuda_graph_min_steps", 64):
                self.record["reason"] = "too_few_steps"
            elif any(p.is_sparse or not p.is_leaf for p in self.parameters):
                self.record["reason"] = "non_leaf_or_sparse_parameters"
            elif any(not isinstance(p, torch.nn.Parameter) for p in self.parameters):
                self.record["reason"] = "unregistered_parameter_tensor"
            elif any(
                t.requires_grad and id(t) not in {id(p) for p in self.parameters}
                for t in self.inputs
                if isinstance(t, torch.Tensor)
            ):
                self.record["reason"] = "untracked_input_gradient"
            elif any(
                p.requires_grad and id(p) not in {id(t) for t in self.parameters}
                for module in self.modules
                for p in module.parameters()
            ):
                self.record["reason"] = "untracked_module_parameters"
            else:
                self.record.update(status="pending", reason=None)

    def _buffers(self):
        seen = set()
        result = []
        for module in self.modules:
            for buffer in module.buffers():
                if id(buffer) not in seen:
                    seen.add(id(buffer))
                    result.append(buffer)
        return tuple(result)

    def _signature(self):
        tensors = (
            self.parameters
            + self._buffers()
            + tuple(
                t for t in self.inputs if isinstance(t, torch.Tensor) and t.layout == torch.strided
            )
        )
        tensor_signature = tuple(
            (id(t), t.data_ptr(), tuple(t.shape), t.dtype, t.device, t.requires_grad)
            for t in tensors
        )
        modes = tuple(
            (id(m), m.training, getattr(m, "p", None), getattr(m, "dropout", None))
            for root in self.modules
            for m in root.modules()
        )
        return tensor_signature, modes

    def _state(self):
        return {
            "parameters": [p.detach().clone() for p in self.parameters],
            "buffers": [b.detach().clone() for b in self._buffers()],
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device),
        }

    def _restore(self, state, gradients=None):
        with torch.no_grad():
            for parameter, value in zip(self.parameters, state["parameters"]):
                parameter.copy_(value)
            for buffer, value in zip(self._buffers(), state["buffers"]):
                buffer.copy_(value)
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        for index, parameter in enumerate(self.parameters):
            parameter.grad = None if gradients is None else gradients[index]

    @staticmethod
    def _flat(output):
        return output if isinstance(output, tuple) else (output,)

    def _execute(self):
        output = self.function()
        loss = self._flat(output)[-1]
        loss.backward()
        return output

    def _replay(self):
        for parameter in self.parameters:
            parameter.grad = None
        loss = self.graph()
        loss.backward()
        self.outputs = (
            tuple(t.detach() for t in self.wrapper.last_output[:-1]) + (loss,)
            if isinstance(self.wrapper.last_output, tuple)
            else loss
        )
        self.gradients = tuple(p.grad for p in self.parameters)
        self.record["replays"] += 1
        return self.outputs

    def _capture(self):
        original_stream = torch.cuda.current_stream(self.device)
        initial = self._state()
        # One real eager update's forward/backward is our reference. No optimizer
        # update is performed during warmup, capture or verification.
        # A first reference backward on the legacy stream creates AccumulateGrad
        # nodes there and can make later capture illegal. Initialize the tape on
        # a side stream, just like native warmup. This adds no optimizer updates.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as preflight:
                reference = self._execute()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        reference_values = tuple(t.detach().clone() for t in self._flat(reference))
        reference_gradients = tuple(
            None if p.grad is None else p.grad.detach().clone() for p in self.parameters
        )
        final = self._state()
        unsafe = {
            "aten::item",
            "aten::_local_scalar_dense",
            "aten::nonzero",
            "aten::_linalg_check_errors",
        }
        forbidden = sorted(unsafe.intersection(row.key for row in preflight.key_averages()))
        if forbidden:
            self.record.update(status="eager", reason="unsafe_operations: " + ", ".join(forbidden))
            self.record["eager_steps"] += 1
            return reference
        started = time.perf_counter()
        try:
            self._restore(initial)
            wrapper = _LossModule(self.function, self.parameters)
            self.wrapper = wrapper
            self.graph = torch.cuda.make_graphed_callables(
                wrapper, (), num_warmup_iters=self.warmups, allow_unused_input=True
            )
            torch.cuda.synchronize(self.device)
            self.record["capture_seconds"] += time.perf_counter() - started
            verification_started = time.perf_counter()
            self._restore(initial)
            candidate = self._replay()
            candidate_state = self._state()
            equal = all(
                torch.equal(left, right)
                for left, right in zip(self._flat(candidate), reference_values)
            )
            equal = equal and all(
                (left is None and right is None)
                or (left is not None and right is not None and torch.equal(left, right))
                for left, right in zip(self.gradients, reference_gradients)
            )
            equal = equal and all(
                torch.equal(left, right)
                for key in ("parameters", "buffers")
                for left, right in zip(candidate_state[key], final[key])
            )
            equal = equal and all(
                torch.equal(candidate_state[key], final[key]) for key in ("cpu_rng", "cuda_rng")
            )
            self.record["verification_seconds"] += time.perf_counter() - verification_started
            if not equal:
                raise ValueError("nonidentical_forward_backward_state_or_rng")
            self.signature = self._signature()
            self.record.update(status="captured", reason=None)
            return candidate
        except (RuntimeError, ValueError) as error:
            contexts = []
            current = error
            while current is not None:
                contexts.append(f"{type(current).__name__}: {str(current).splitlines()[0]}")
                current = current.__context__
            self.record["errors"] = contexts
            self.record["capture_seconds"] = max(
                self.record["capture_seconds"], time.perf_counter() - started
            )
            if isinstance(error, getattr(torch, "AcceleratorError", ())):
                torch.cuda.set_stream(original_stream)
                self.record.update(status="failed", reason="cuda_capture_error_requires_restart")
                raise RuntimeError(
                    "CUDA graph capture failed at the driver/runtime level. Restart this process "
                    "with --no_cuda_graph; continuing could corrupt CUDA RNG/stream state."
                ) from error
            self._restore(final, reference_gradients)
            self.graph, self.outputs, self.gradients = None, None, ()
            self.record.update(
                status="eager", reason=f"capture_rejected: {str(error).splitlines()[0]}"
            )
            self.record["eager_steps"] += 1
            return reference

    def __call__(self):
        output = self._run()
        if getattr(self, "trace", False):
            self.record.setdefault("loss_trace", []).append(float(self._flat(output)[-1].detach()))
        return output

    def _run(self):
        if self.graph is not None:
            if self.signature == self._signature():
                return self._replay()
            self.graph, self.outputs, self.gradients = None, None, ()
            self.record["invalidations"] += 1
            self.record.update(status="pending", reason="parameter_storage_or_mode_changed")
        if self.record["status"] == "pending":
            return self._capture()
        self.record["eager_steps"] += 1
        return self._execute()


class CUDAGraphCallable:
    """Autograd-aware partial capture for deterministic pure kernel functions.

    Dense tensor inputs may change values, not shape/dtype/stride/grad contract.
    Solves, sampling and optimizers remain outside the callable. No double backward.
    """

    def __init__(self, function, args, stage, steps):
        self.function = function
        self.args = args
        self.graphed = None
        self.signature = None
        self.record = {
            "stage": stage,
            "requested": bool(getattr(args, "cuda_graph", False)),
            "status": "pending",
            "reason": None,
            "capture_seconds": 0.0,
            "replays": 0,
            "invalidations": 0,
        }
        if not self.record["requested"]:
            self.record.update(status="eager", reason="not_requested")
        elif steps < getattr(args, "cuda_graph_min_steps", 64):
            self.record.update(status="eager", reason="too_few_steps")
        if self.record["requested"]:
            if not hasattr(args, "_cuda_graph_records"):
                args._cuda_graph_records = []
            args._cuda_graph_records.append(self.record)

    def __call__(self, *inputs):
        if not inputs or any(
            not isinstance(t, torch.Tensor) or t.device.type != "cuda" or t.layout != torch.strided
            for t in inputs
        ):
            self.record.update(status="eager", reason="non_dense_or_cpu_input")
        if self.record["status"] == "eager":
            return self.function(*inputs)
        signature = tuple((t.shape, t.stride(), t.dtype, t.device, t.requires_grad) for t in inputs)
        if self.graphed is not None and self.signature == signature:
            self.record["replays"] += 1
            return self.graphed(*inputs)
        if self.graphed is not None:
            self.graphed = None
            self.record["invalidations"] += 1
        differentiable = tuple(t for t in inputs if t.requires_grad)
        if not differentiable:
            return self.function(*inputs)
        initial_cpu_rng = torch.get_rng_state()
        device = inputs[0].device
        initial_cuda_rng = torch.cuda.get_rng_state(device)

        # Retain the eager graph so a rejected capture can return it unchanged.
        def derivatives(output):
            tensors = CUDAGraphStep._flat(output)
            scalar = sum(t.square().sum() / t.numel() for t in tensors)
            return torch.autograd.grad(scalar, differentiable, retain_graph=True, allow_unused=True)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            reference = self.function(*inputs)
            reference_gradients = derivatives(reference)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        reference_values = CUDAGraphStep._flat(reference)
        started = time.perf_counter()
        original_stream = torch.cuda.current_stream(device)
        try:
            self.graphed = torch.cuda.make_graphed_callables(
                self.function,
                inputs,
                num_warmup_iters=getattr(self.args, "cuda_graph_warmup", 3),
                allow_unused_input=True,
            )
            torch.set_rng_state(initial_cpu_rng)
            torch.cuda.set_rng_state(initial_cuda_rng, device)
            candidate = self.graphed(*inputs)
            candidate_gradients = derivatives(candidate)
            equal = all(
                torch.equal(x, y) for x, y in zip(reference_values, CUDAGraphStep._flat(candidate))
            )
            equal = equal and all(
                (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
                for x, y in zip(reference_gradients, candidate_gradients)
            )
            if not equal:
                raise ValueError("nonidentical_kernel_or_gradient")
            self.record.update(status="captured", reason=None)
            self.signature = signature
            self.record["replays"] += 1
            # Native forward/backward graphs share a private memory pool.
            # Derivative verification may overwrite forward intermediates.
            # Start a fresh forward before the caller's real GP backward.
            return self.graphed(*inputs)
        except (RuntimeError, ValueError) as error:
            self.graphed = None
            if isinstance(error, getattr(torch, "AcceleratorError", ())):
                torch.cuda.set_stream(original_stream)
                self.record.update(status="failed", reason="cuda_capture_error_requires_restart")
                raise RuntimeError(
                    "CUDA kernel capture failed at the driver/runtime level. "
                    "Restart this process with --no_cuda_graph."
                ) from error
            self.record.update(
                status="eager", reason=f"capture_rejected: {str(error).splitlines()[0]}"
            )
            return reference
        finally:
            if self.record["status"] != "failed":
                torch.set_rng_state(initial_cpu_rng)
                torch.cuda.set_rng_state(initial_cuda_rng, device)
            self.record["capture_seconds"] += time.perf_counter() - started


def method_acceleration_support():
    """Complete registry coverage, with stage limitations instead of fake speedups."""
    from graphslim.reduction import get_method_spec, list_methods

    reduction = {
        "gcgp": "kernel_pair",
        "clustgdd": "pretrain_and_refine",
        "sfgc": "expert_training_only",
        "gdem": "spectral_loss_and_pretraining",
        "gcsntk": "eager_sparse_kernel_and_checked_solve",
        "geom": "eager_dynamic_curriculum_and_second_order",
        "simgc": "teacher_training_only_dynamic_edges_eager",
        "vng": "embedding_training_and_eager_clustering",
        "kcenter": "embedding_training_and_eager_selection",
        "herding": "embedding_training_and_eager_selection",
    }
    higher_order = {"gcond", "gcondx", "doscond", "doscondx", "sgdd", "msgc"}
    for method in higher_order:
        reduction[method] = "eager_second_order"
    for method in ("gcdm", "gcdmx"):
        reduction[method] = "eager_dynamic_sampling"
    return [
        {
            "method": method,
            "family": get_method_spec(method).family,
            "downstream": "shared_first_order_training",
            "reduction": reduction.get(method, "eager_cpu_or_noniterative"),
            "policy": "opt_in_exact_check_and_eager_fallback",
        }
        for method in list_methods()
    ]

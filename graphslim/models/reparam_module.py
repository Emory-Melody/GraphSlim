# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import torch
import torch.nn as nn
import warnings
import types
from collections import namedtuple
from contextlib import contextmanager


class ReparamModule(nn.Module):
    """Module wrapper exposing all trainable parameters as a differentiable flat tensor."""

    def _get_module_from_name(self, mn):
        """Resolve a dotted submodule path relative to the wrapper.

        :param mn: Dotted path identifying a registered submodule.
        :returns: Submodule at that path, or this wrapper for an empty path.
        """
        if mn == "":
            return self
        m = self
        for p in mn.split("."):
            m = getattr(m, p)
        return m

    def __init__(self, module):
        """Initialize ReparamModule and its configuration.

        :param module: PyTorch module whose parameters will be flattened by this wrapper.
        """
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []  # (module name/path, param name)
        shared_param_memo = {}
        shared_param_infos = []  # (module name/path, param name, src module name/path, src param_name)
        params = []
        param_numels = []
        param_shapes = []
        for mn, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    if p in shared_param_memo:
                        shared_mn, shared_n = shared_param_memo[p]
                        shared_param_infos.append((mn, n, shared_mn, shared_n))
                    else:
                        shared_param_memo[p] = (mn, n)
                        param_infos.append((mn, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())

        assert len(set(p.dtype for p in params)) <= 1, (
            "expects all parameters in module to have same dtype"
        )

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter("flat_param", flat_param)
        self.param_numel = flat_param.numel()
        del params
        del shared_param_memo

        # deregister the names as parameters
        for mn, n in self._param_infos:
            delattr(self._get_module_from_name(mn), n)
        for mn, n, _, _ in self._shared_param_infos:
            delattr(self._get_module_from_name(mn), n)

        # register the views as plain attributes
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        buffer_infos = []
        for mn, m in self.named_modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((mn, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        """Trace forward paths with explicit flattened parameters and buffers using TorchScript.

        :param example_input: Example tensor or tuple of positional inputs for tracing.
        :param trace_kwargs: Keyword options forwarded to ``torch.jit.trace_module``.
        :returns: This wrapper with traced forward helpers installed.
        """
        assert self._traced_self is None, "This ReparamModule is already traced"

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        """Remove temporary tensor views of flattened parameters from wrapped submodules."""
        for mn, n in self._param_infos:
            setattr(self._get_module_from_name(mn), n, None)

    def _apply(self, *args, **kwargs):
        """Apply a tensor conversion to the traced module or the ordinary wrapper.

        :param args: Configuration namespace containing the options used by this operation.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        :returns: This module after device or dtype conversion.
        """
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        """Install shaped parameter views while preserving shared parameter aliases.

        :param flat_param: Flattened model parameters; views preserve gradients through this tensor.
        """
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        for (mn, n), p in zip(self._param_infos, ps):
            setattr(self._get_module_from_name(mn), n, p)  # This will set as plain attr
        for mn, n, shared_mn, shared_n in self._shared_param_infos:
            setattr(
                self._get_module_from_name(mn),
                n,
                getattr(self._get_module_from_name(shared_mn), shared_n),
            )

    @contextmanager
    def unflattened_param(self, flat_param):
        """Temporarily bind a flattened parameter tensor to the wrapped module.

        :param flat_param: Flattened model parameters; views preserve gradients through this tensor.
        :returns: Context manager that restores previous parameter views on exit.
        """
        saved_views = [getattr(self._get_module_from_name(mn), n) for mn, n in self._param_infos]
        self._unflatten_param(flat_param)
        try:
            yield
        finally:
            for (mn, n), p in zip(self._param_infos, saved_views):
                setattr(self._get_module_from_name(mn), n, p)
            for mn, n, shared_mn, shared_n in self._shared_param_infos:
                setattr(
                    self._get_module_from_name(mn),
                    n,
                    getattr(self._get_module_from_name(shared_mn), shared_n),
                )

    @contextmanager
    def replaced_buffers(self, buffers):
        """Temporarily replace registered buffers during a functional model evaluation.

        :param buffers: Replacement model buffers in the wrapper registration order.
        :returns: Context manager that restores previous buffers on exit.
        """
        for (mn, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(self._get_module_from_name(mn), n, new_b)
        try:
            yield
        finally:
            for mn, n, old_b in self._buffer_infos:
                setattr(self._get_module_from_name(mn), n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        """Evaluate the wrapped model with explicit parameters and replacement buffers.

        :param flat_param: Flattened model parameters; views preserve gradients through this tensor.
        :param buffers: Replacement model buffers in the wrapper registration order.
        :param inputs: Positional inputs forwarded to the wrapped model.
        :param kwinputs: Keyword inputs forwarded to the wrapped model.
        :returns: Output returned by the wrapped module.
        """
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        """Evaluate the wrapped model with explicit flattened parameters.

        :param flat_param: Flattened model parameters; views preserve gradients through this tensor.
        :param inputs: Positional inputs forwarded to the wrapped model.
        :param kwinputs: Keyword inputs forwarded to the wrapped model.
        :returns: Output returned by the wrapped module.
        """
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        """Run a functional forward pass with optional replacement parameters and buffers.

        :param flat_param: Flattened model parameters; views preserve gradients through this tensor. Defaults to ``None``, which uses the wrapper's own ``flat_param``.
        :param buffers: Replacement model buffers in the wrapper registration order. Defaults to ``None``.
        :param inputs: Positional inputs forwarded to the wrapped model.
        :param kwinputs: Keyword inputs forwarded to the wrapped model.
        :returns: Output returned by the wrapped module; gradients flow through the chosen parameter tensor.
        """
        if flat_param is None:
            flat_param = self.flat_param
        # Accept row/column vectors while keeping one-element parameters 1-D.
        flat_param = flat_param.reshape(-1)
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(
                flat_param, tuple(buffers), *inputs, **kwinputs
            )

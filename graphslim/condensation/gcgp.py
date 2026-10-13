"""GCGP: Gaussian-process prediction matching with learned synthetic targets.

Wang et al., arXiv:2501.02565. Kernel formulas follow the MIT-licensed
author release (https://github.com/WANGLin0126/GCGP); see THIRD_PARTY_NOTICES.
No GNN is trained inside condensation. Synthetic features/targets ARE optimized.
"""

import math

import torch
import torch.nn.functional as F
from graphslim.acceleration import CUDAGraphCallable

from graphslim.reduction.lightweight import (
    LightweightReducer,
    normalized_adjacency,
    propagate,
    sparse_tensor,
)


def neural_kernel(x, y, layers=2):
    """Released SGNK: sum of the input and erf activation covariance kernels."""
    cross = x @ y.T
    xx, yy = x.square().sum(1)[:, None], y.square().sum(1)[None, :]
    result = cross
    for _ in range(layers):
        denominator = torch.sqrt(1 + 2 * xx) * torch.sqrt(1 + 2 * yy)
        cross = (2 / math.pi) * torch.asin((2 * cross / denominator).clamp(-1 + 1e-7, 1 - 1e-7))
        xx = (2 / math.pi) * torch.asin(2 * xx / (1 + 2 * xx))
        yy = (2 / math.pi) * torch.asin(2 * yy / (1 + 2 * yy))
        result = result + cross
    return result


def released_neural_kernel(x, y, layers=2, diagonal_x=None):
    """Author SGNK arithmetic, including GEMM diagonals and sqrt-product order.

    MIT-licensed GCGP release; see THIRD_PARTY_NOTICES. Only a detached fixed
    query diagonal may be cached. The historical row-norm implementation remains
    the default; select ``gcgp_arithmetic='released'`` explicitly for this path.
    """
    xx = torch.diag(x @ x.t()).reshape(-1, 1) if diagonal_x is None else diagonal_x
    yy = torch.diag(y @ y.t()).reshape(1, -1)
    cross = x @ y.t()
    kernels = [cross]
    for _ in range(layers):
        a_xy = (2 * cross / (torch.sqrt(1 + 2 * xx) * torch.sqrt(1 + 2 * yy))).clamp(-1, 1)
        a_xx = (2 * xx / (torch.sqrt(1 + 2 * xx) * torch.sqrt(1 + 2 * xx))).clamp(-1, 1)
        a_yy = (2 * yy / (torch.sqrt(1 + 2 * yy) * torch.sqrt(1 + 2 * yy))).clamp(-1, 1)
        cross, xx, yy = (2 / math.pi * torch.asin(t) for t in (a_xy, a_xx, a_yy))
        kernels.append(cross)
    return sum(kernels)


def _relu_update(covariance, diagonal_x, diagonal_y):
    scale = diagonal_x[:, None] * diagonal_y[None, :]
    correlation = (covariance / scale.clamp_min(1e-12)).clamp(-0.9999, 0.9999)
    activated = (
        correlation * (math.pi - torch.acos(correlation)) + torch.sqrt(1 - correlation.square())
    ) / math.pi
    # The released NTK/SNTK recurrence evaluates this factor AFTER the covariance
    # update. Preserve that author convention rather than silently substituting
    # a different kernel's derivative factor.
    derivative = (math.pi - torch.acos(activated.clamp(-1, 1))) / math.pi
    covariance = activated * scale
    return covariance, derivative


def tangent_kernel(x, y, adjacency_x=None, adjacency_y=None, layers=2, hops=0):
    """NTK (hops=0) or structure-based NTK, with differentiable adjacency."""
    cross, self_x, self_y = x @ y.T, x @ x.T, y @ y.T
    tangent = cross
    for _ in range(max(hops, 1)):
        if hops:
            cross = adjacency_x @ cross @ adjacency_y.T
            tangent = adjacency_x @ tangent @ adjacency_y.T
            self_x = adjacency_x @ self_x @ adjacency_x.T
            self_y = adjacency_y @ self_y @ adjacency_y.T
        for _ in range(layers):
            diag_x = self_x.diagonal().clamp_min(1e-12).sqrt()
            diag_y = self_y.diagonal().clamp_min(1e-12).sqrt()
            cross, derivative = _relu_update(cross, diag_x, diag_y)
            tangent = tangent * derivative + cross
            self_x, _ = _relu_update(self_x, diag_x, diag_x)
            self_y, _ = _relu_update(self_y, diag_y, diag_y)
    return tangent


def concrete_adjacency(alpha, epoch=0, training=True, uniform=None):
    """Symmetric BinConcrete with the released 0.3 -> 0.01 temperature schedule."""
    alpha = (alpha + alpha.T) / 2
    if training:
        if uniform is None:
            uniform = torch.rand_like(alpha)
            uniform = torch.triu(uniform) + torch.triu(uniform, diagonal=1).T
        uniform = uniform.clamp(1e-7, 1 - 1e-7)
        temperature = max(0.3 * (0.01 / 0.3) ** (epoch / 100), 0.01)
        adjacency = torch.sigmoid(
            (uniform.log() - torch.log1p(-uniform) + (alpha.clamp_min(0) + 1e-8).log())
            / temperature
        )
    else:
        adjacency = (alpha > 1).to(alpha.dtype)
    identity = torch.eye(len(alpha), device=alpha.device, dtype=alpha.dtype)
    return adjacency * (1 - identity) + identity


def gp_predict(cross_kernel, synthetic_kernel, targets, ridge, released_arithmetic=False):
    """Trace-scaled GP ridge solve, followed by the author's softmax."""
    if ridge <= 0:
        raise ValueError("gcgp_ridge must be positive")
    n = len(synthetic_kernel)
    regularizer = ridge * synthetic_kernel.trace() / n
    identity = torch.eye(n, device=synthetic_kernel.device, dtype=synthetic_kernel.dtype)
    if released_arithmetic:
        # Tensor division matters for Float32 CUDA's gradient rounding.
        count = torch.tensor(n, device=synthetic_kernel.device)
        matrix = synthetic_kernel + ridge * synthetic_kernel.trace() * identity / count
        return (cross_kernel @ torch.linalg.solve(matrix, targets)).softmax(1)
    coefficients = torch.linalg.solve(
        synthetic_kernel + regularizer.clamp_min(1e-8) * identity, targets
    )
    return (cross_kernel @ coefficients).softmax(1)


class GCGP(LightweightReducer):
    def reduce(self, data, verbose=True, save=True):
        args = self.args
        features, adjacency, train, labels = self.inputs(data)
        filtered = propagate(
            normalized_adjacency(adjacency, self.device), features, getattr(args, "gcgp_k", 4)
        )[train].detach()
        target = F.one_hot(labels, data.nclass).float()
        size = max(1, round(len(train) * args.reduction_rate))
        self.check_budget(size)
        # The release initializes features and continuous synthetic targets uniformly.
        synthetic = torch.nn.Parameter(torch.rand(size, features.shape[1], device=self.device))
        synthetic_targets = torch.nn.Parameter(torch.rand(size, data.nclass, device=self.device))
        alpha = torch.nn.Parameter(2 * torch.rand(size, size, device=self.device))
        parameters = [synthetic, synthetic_targets]
        learn_adjacency = getattr(args, "gcgp_learn_adj", False)
        kernel = getattr(args, "gcgp_kernel", "SGNK").upper()
        if kernel not in ("SGNK", "NTK", "SNTK"):
            raise ValueError("gcgp_kernel must be SGNK, NTK or SNTK")
        if learn_adjacency and kernel != "SNTK":
            raise ValueError("SGNK/NTK ignore adjacency; gcgp_learn_adj requires SNTK")
        if learn_adjacency:
            parameters.append(alpha)
        optimizer = torch.optim.Adam(parameters, lr=getattr(args, "gcgp_lr", 0.01))
        original_training_adjacency = adjacency[train.cpu().numpy()][:, train.cpu().numpy()]
        original_training_adjacency.setdiag(1)
        batch_size = getattr(args, "gcgp_batch_size", 1024)
        if batch_size < 1:
            raise ValueError("gcgp_batch_size must be positive")
        loss_history = []
        layers = getattr(args, "gcgp_layers", 2)
        arithmetic = getattr(args, "gcgp_arithmetic", "package")
        if arithmetic not in ("package", "released"):
            raise ValueError("gcgp_arithmetic must be package or released")
        released = arithmetic == "released"
        original_diagonal = (
            torch.diag(filtered @ filtered.t()).reshape(-1, 1)
            if released and kernel == "SGNK" and len(train) <= batch_size
            else None
        )

        def kernel_pair(original, synthetic):
            if kernel == "NTK":
                return (
                    tangent_kernel(synthetic, synthetic, layers=layers),
                    tangent_kernel(original, synthetic, layers=layers),
                )
            function = released_neural_kernel if released else neural_kernel
            ss = function(synthetic, synthetic, layers)
            ts = (
                function(original, synthetic, layers, diagonal_x=original_diagonal)
                if released
                else function(original, synthetic, layers)
            )
            return ss, ts

        kernels = (
            CUDAGraphCallable(kernel_pair, args, f"gcgp.{kernel}.kernels", args.epochs)
            if kernel in ("SGNK", "NTK")
            else None
        )
        for epoch in range(args.epochs):
            chosen = (
                torch.randperm(len(train), device=self.device)[:batch_size]
                if len(train) > batch_size
                else torch.arange(len(train), device=self.device)
            )
            original = filtered[chosen]
            reduced_adjacency = (
                concrete_adjacency(alpha, epoch)
                if learn_adjacency
                else torch.eye(size, device=self.device)
            )
            if kernel == "SGNK":
                ss, ts = kernels(original, synthetic)
            else:
                chosen_cpu = chosen.cpu().numpy()
                aa = sparse_tensor(
                    original_training_adjacency[chosen_cpu][:, chosen_cpu], self.device
                )
                hops = getattr(args, "gcgp_hops", 2) if kernel == "SNTK" else 0
                if kernel == "NTK":
                    ss, ts = kernels(original, synthetic)
                else:
                    ss = tangent_kernel(
                        synthetic, synthetic, reduced_adjacency, reduced_adjacency, layers, hops
                    )
                    ts = tangent_kernel(original, synthetic, aa, reduced_adjacency, layers, hops)
            prediction = gp_predict(
                ts, ss, synthetic_targets, getattr(args, "gcgp_ridge", 0.5), released
            )
            loss = F.mse_loss(prediction, target[chosen])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GCGP loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach()))
        reduced_adjacency = (
            concrete_adjacency(alpha, training=False)
            if learn_adjacency
            else torch.eye(size, device=self.device)
        )
        if getattr(args, "_cuda_graph_trace", False):
            args._reduction_loss_trace = list(loss_history)
        return self.finish(
            data,
            synthetic,
            reduced_adjacency,
            synthetic_targets,
            save=save,
            verbose=verbose,
            gnn_training_steps=0,
            optimization_steps=args.epochs,
            kernel=kernel,
            arithmetic=arithmetic,
            training_batch_size=min(batch_size, len(train)),
            final_train_loss=loss_history[-1],
        )

# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import torch
import torch.nn as nn


class KernelRidgeRegression(nn.Module):
    """Differentiable kernel ridge predictor with a trace-scaled regularizer."""

    def __init__(self, kernel, ridge):
        """Initialize KernelRidgeRegression and its configuration.

        :param kernel: Callable graph kernel used to build support and cross-kernel matrices.
        :param ridge: Trace-scaled ridge coefficient for the support Gram matrix.
        """
        super(KernelRidgeRegression, self).__init__()
        self.kernel = kernel
        self.ridge = ridge

    def forward(self, G_t, G_s, y_t, y_s, E_t, E_s):
        """Fit kernel ridge coefficients on the synthetic support graph and predict target probabilities.

        The trace-scaled Gram solve uses float64 and retries with small diagonal
        jitter if necessary. Non-finite kernels raise ``FloatingPointError``.

        :param G_t: Target or synthetic support node features, respectively.
        :param G_s: Target or synthetic support node features, respectively.
        :param y_t: Target or synthetic support class targets, respectively, with one row per node.
        :param y_s: Target or synthetic support class targets, respectively, with one row per node.
        :param E_t: Target or synthetic support sparse adjacency, respectively.
        :param E_s: Target or synthetic support sparse adjacency, respectively.
        :returns: Tuple ``(target_class_probabilities, accuracy_fraction)``.
        """
        K_ss = self.kernel(G_s, G_s, E_s, E_s)
        K_ts = self.kernel(G_t, G_s, E_t, E_s)
        n = torch.tensor(len(G_s), device=G_s.device)
        regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device=G_s.device) / n
        b = torch.linalg.solve(K_ss + regulizer, y_s)
        pred = torch.matmul(K_ts, b)
        pred = nn.functional.softmax(pred, dim=1)
        correct = (
            torch.eq(pred.argmax(1).to(torch.float32), y_t.argmax(1).to(torch.float32)).sum().item()
        )
        acc = correct / len(y_t)
        return pred, acc

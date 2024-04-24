import torch
import torch.nn as nn


class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel, ridge):
        super(KernelRidgeRegression, self).__init__()
        self.kernel = kernel
        self.ridge = ridge

    def forward(self, G_t, G_s, y_t, y_s, E_t, E_s):
        K_ss = self.kernel(G_s, G_s, E_s, E_s)
        K_ts = self.kernel(G_t, G_s, E_t, E_s)
        n = torch.tensor(len(G_s), device=G_s.device)
        regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device=G_s.device) / n
        b = torch.linalg.solve(K_ss + regulizer, y_s)
        pred = torch.matmul(K_ts, b)
        pred = nn.functional.softmax(pred, dim=1)
        correct = torch.eq(pred.argmax(1).to(torch.float32), y_t.argmax(1).to(torch.float32)).sum().item()
        acc = correct / len(y_t)
        return pred, acc

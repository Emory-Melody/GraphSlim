from itertools import product

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn


# from graph import mx_inv, mx_inv_sqrt, mx_tr

def mx_inv(mx, device):
    if isinstance(mx, torch_sparse.tensor.SparseTensor):
        mx = mx.to_dense()
    elif isinstance(mx, scipy.sparse.csr.csr_matrix):
        mx = torch.FloatTensor(mx.toarray()).to(device)
    U, D, V = torch.svd(mx)
    eps = 0.009
    D_min = torch.min(D)
    if D_min < eps:
        D_1 = torch.zeros_like(D)
        D_1[D > D_min] = 1 / D[D > D_min]
    else:
        D_1 = 1 / D
    # D_1 = 1 / D #.clamp(min=0.005)

    return U @ D_1.diag() @ V.t()


def mx_inv_sqrt(mx):
    # singular values need to be distinct for backprop
    U, D, V = torch.svd(mx)
    D_min = torch.min(D)
    eps = 0.009
    if D_min < eps:
        D_1 = torch.zeros_like(D)
        D_1[D > D_min] = 1 / D[D > D_min]  # .sqrt()
    else:
        D_1 = 1 / D  # .sqrt()
    # D_1 = 1 / D.clamp(min=0.005).sqrt()
    return U @ D_1.sqrt().diag() @ V.t(), U @ D_1.diag() @ V.t()

def mx_tr(mx):
    return mx.diag().sum()


def get_mgrid(sidelen, dim=2):
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords.to(torch.float32)


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class EdgeBlock(nn.Module):
    def __init__(self, in_, out_, dtype=torch.float32) -> None:
        super(EdgeBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_, out_, dtype=dtype),
            nn.BatchNorm1d(out_, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GraphonLearner(nn.Module):
    def __init__(
            self,
            node_feature,
            nfeat=256,
            nnodes=50,
            device="cuda",
            args={},
            num_hidden_layers=3,
            **kwargs
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.step_size = nnodes
        self.ep_ratio = args.ep_ratio
        self.sinkhorn_iter = args.sinkhorn_iter
        self.mx_size = args.mx_size

        self.edge_index = np.array(
            list(product(range(self.step_size), range(self.step_size)))
        ).T

        self.net0 = nn.ModuleList(
            [
                EdgeBlock(node_feature * 2, nfeat),
                EdgeBlock(nfeat, nfeat),
                nn.Linear(nfeat, 1, dtype=torch.float32),
            ]
        )

        self.net1 = nn.ModuleList(
            [
                EdgeBlock(2, nfeat),
                EdgeBlock(nfeat, nfeat),
                nn.Linear(nfeat, 1, dtype=torch.float32),
            ]
        )

        self.P = nn.Parameter(
            torch.Tensor(self.mx_size, self.step_size).to(torch.float32).uniform_(0, 1)
        )  # transport plan
        self.Lx_inv = None

        self.output = nn.Linear(nfeat, 1)
        self.act = F.relu
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

        self.apply(weight_reset)

    def forward(self, c, inference=False, Lx=None):
        if inference == True:
            self.eval()
        else:
            self.train()
        x0 = get_mgrid(c.shape[0]).to(self.device)
        c = torch.cat([c[self.edge_index[0]], c[self.edge_index[1]]], axis=1)
        for layer in range(len(self.net0)):
            c = self.net0[layer](c)
            if layer == 0:
                x = self.net1[layer](x0)
            else:
                x = self.net1[layer](x)

            if layer != (len(self.net0) - 1):
                # use node feature to guide the graphon generating process
                x = x * c
            else:
                x = (1 - self.ep_ratio) * x + self.ep_ratio * c

        # x = self.output(x)
        # adj = self.output(x).reshape(self.step_size, self.step_size)
        adj = x.reshape(self.step_size, self.step_size)

        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))

        if inference == True:
            return adj
        if Lx is not None and self.Lx_inv is None:
            self.Lx_inv = mx_inv(Lx, c.device)
        # try:
        #     opt_loss = self.opt_loss(adj)
        # except Exception as e:
        #     print(e)
        #     opt_loss = torch.tensor(0).to(self.device)
        opt_loss = self.opt_loss(adj)
        return adj, opt_loss

    def opt_loss(self, adj):
        Ly_inv_rt, Ly_inv = mx_inv_sqrt(adj)
        m = self.step_size
        P = self.P.abs()

        for _ in range(self.sinkhorn_iter):
            P = P / P.sum(dim=1, keepdim=True)
            P = P / P.sum(dim=0, keepdim=True)

        # if self.args.use_symeig:
        # sqrt = torch.symeig(
        sqrt = torch.linalg.eigh(
            Ly_inv_rt @ self.P.t() @ self.Lx_inv @ self.P @ Ly_inv_rt
        )
        loss = torch.abs(
            mx_tr(Ly_inv) * m - 2 * torch.sqrt(sqrt[0].clamp(min=2e-20)).sum()
        )
        return loss

    @torch.no_grad()
    def inference(self, c):
        return self.forward(c, inference=True)

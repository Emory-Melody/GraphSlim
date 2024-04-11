import copy

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from pygsp import graphs
from torch import tensor
from torch_sparse import SparseTensor
from tqdm import trange

from graphslim.coarsening.utils import contract_variation_edges, contract_variation_linear, get_proximity_measure, \
    matching_optimal, matching_greedy, get_coarsening_matrix, coarsen_matrix, coarsen_vector, zero_diag
from graphslim.dataset.convertor import pyg2gsp
from graphslim.dataset.utils import splits
from graphslim.models import GCN
from graphslim.utils import one_hot


def router_coarse(data, args):
    return CoarseningBase(data, args)


class CoarseningBase:
    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

    def train(self):
        args = self.args
        data = self.data
        device = self.device

        cpu_data = copy.deepcopy(data)

        candidate, C_list, Gc_list = coarsening(pyg2gsp(data), 1 - args.reduction_rate, args.method)
        model = GCN(data.x.shape[1], args.hidden, data.nclass, lr=args.lr, weight_decay=args.weight_decay,
                    device=device).to(device)
        all_acc = []

        for i in trange(args.runs):
            coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = process_coarsened(
                cpu_data, candidate, C_list, Gc_list)
            coarsen_features = coarsen_features.to(device)
            coarsen_train_labels = coarsen_train_labels.to(device)
            coarsen_train_mask = coarsen_train_mask.to(device)
            coarsen_val_labels = coarsen_val_labels.to(device)
            coarsen_val_mask = coarsen_val_mask.to(device)
            coarsen_edge = SparseTensor(row=coarsen_edge[1], col=coarsen_edge[0]).to(device)
            data = splits(data, args.split)
            data = data.to(device)

            if args.normalize_features:
                coarsen_features = F.normalize(coarsen_features, p=1)
                data.x = F.normalize(data.x, p=1)

            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_val_loss = float('inf')
            val_loss_history = []
            for epoch in range(args.epochs):

                model.train()
                optimizer.zero_grad()
                out = model(coarsen_features, coarsen_edge)
                loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                loss.backward()
                optimizer.step()

                model.eval()
                pred = model(coarsen_features, coarsen_edge)
                val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    pred = model(data.x, data.sparse_adj).max(1)[1]
                    test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(
                        data.test_mask.sum())
                    all_acc.append(test_acc)

                # val_loss_history.append(val_loss)
                if args.early_stopping > 0:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        break

        print('ave_test_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))


def coarsening(H, coarsening_ratio, coarsening_method):
    if H.A.shape[-1] != H.A.shape[1]:
        H.logger.error('Inconsistent shape to extract components. '
                       'Square matrix required.')
        return None

    if H.is_directed():
        raise NotImplementedError('Directed graphs not supported yet.')

    graphs = []

    visited = np.zeros(H.A.shape[-1], dtype=bool)

    while not visited.all():
        stack = set([np.nonzero(~visited)[0][0]])
        comp = []

        while len(stack):
            v = stack.pop()
            if not visited[v]:
                comp.append(v)
                visited[v] = True

                stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                  if not visited[idx]]))

        comp = sorted(comp)
        G = H.subgraph(comp)
        G.info = {'orig_idx': comp}
        graphs.append(G)

    print('the number of subgraphs is', len(graphs))
    candidate = sorted(graphs, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list = []
    Gc_list = []
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    return candidate, C_list, Gc_list


def process_coarsened(data, candidate, C_list, Gc_list):
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    n_classes = max(data.y) + 1
    features = data.x
    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask) + torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat(
                [coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat(
                [coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask) + torch.sum(H_val_mask) > 1:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    # print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.from_numpy(np.array([coarsen_row, coarsen_col])).long()
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge


def coarsen(
        G,
        K=10,
        r=0.5,
        max_levels=10,
        method="variation_neighborhood",
        algorithm="greedy",
        Uk=None,
        lk=None,
        max_level_r=0.99,
):
    """
    This function provides a common interface for coarsening algorithms that contract subgraphs

    Parameters
    ----------
    G : pygsp Graph
    K : int
        The size of the subspace we are interested in preserving.
    r : float between (0,1)
        The desired reduction defined as 1 - n/N.
    method : String
        ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']

    Returns
    -------
    C : np.array of size n x N
        The coarsening matrix.
    Gc : pygsp Graph
        The smaller graph.
    Call : list of np.arrays
        Coarsening matrices for each level
    Gall : list of (n_levels+1) pygsp Graphs
        All graphs involved in the multilevel coarsening

    Example
    -------
    C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
    """
    r = np.clip(r, 0, 0.999)
    G0 = G
    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    C = sp.sparse.eye(N, format="csc")
    Gc = G

    Call, Gall = [], []
    Gall.append(G)

    for level in range(1, max_levels + 1):

        G = Gc

        # how much more we need to reduce the current graph
        r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

        if "variation" in method:

            if level == 1:
                if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                    mask = lk < 1e-10
                    lk[mask] = 1
                    lsinv = lk ** (-0.5)
                    lsinv[mask] = 0
                    B = Uk[:, :K] @ np.diag(lsinv[:K])
                else:
                    offset = 2 * max(G.dw)
                    T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                    lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                    lk = (offset - lk)[::-1]
                    Uk = Uk[:, ::-1]
                    mask = lk < 1e-10
                    lk[mask] = 1
                    lsinv = lk ** (-0.5)
                    lsinv[mask] = 0
                    B = Uk @ np.diag(lsinv)
                A = B
            else:
                B = iC.dot(B)
                d, V = np.linalg.eig(B.T @ (G.L).dot(B))
                mask = d == 0
                d[mask] = 1
                dinvsqrt = (d + 1e-9) ** (-1 / 2)
                dinvsqrt[mask] = 0
                A = B @ np.diag(dinvsqrt) @ V

            if method == "variation_edges":
                coarsening_list = contract_variation_edges(
                    G, K=K, A=A, r=r_cur, algorithm=algorithm
                )
            else:
                coarsening_list = contract_variation_linear(
                    G, K=K, A=A, r=r_cur, mode=method
                )

        else:
            weights = get_proximity_measure(G, method, K=K)

            if algorithm == "optimal":
                # the edge-weight should be light at proximal edges
                weights = -weights
                if "rss" not in method:
                    weights -= min(weights)
                coarsening_list = matching_optimal(G, weights=weights, r=r_cur)

            elif algorithm == "greedy":
                coarsening_list = matching_greedy(G, weights=weights, r=r_cur)

        iC = get_coarsening_matrix(G, coarsening_list)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes

        C = iC.dot(C)
        Call.append(iC)

        Wc = zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = graphs.Graph(Wc)
        else:
            Gc = graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
        Gall.append(Gc)

        n = Gc.N

        if n <= n_target:
            break

    return C, Gc, Call, Gall

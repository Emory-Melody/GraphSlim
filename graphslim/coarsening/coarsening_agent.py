import copy
import torch
import torch.nn.functional as F
from torch import tensor
from torch_sparse import SparseTensor
from tqdm import trange
import numpy as np

from graphslim.dataset.convertor import pyg2gsp
from graphslim.models import GCN
from graphslim.coarsening.coarsening_utils import coarsening, process_coarsened
from graphslim.dataset.utils import splits

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

        candidate, C_list, Gc_list = coarsening(pyg2gsp(data), 1 - args.reduction_rate, args.coarsening_method)
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
            data = splits(data, data.nclass, args.experiment)
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

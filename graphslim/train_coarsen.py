import argparse
import copy
import os

from tqdm import trange
from torch import tensor
from torch_sparse import SparseTensor
from dataset.convertor import pyg2gsp

from coarsening import *
from dataset import *
from models import GCN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed')
    # TODO: implement setting
    parser.add_argument('--setting', '-S', type=str, default='trans', help='trans/ind')
    parser.add_argument('--experiment', type=str, default='fixed')  # 'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--normalize_features', type=bool, default=True)
    # remind: 0.026 equals 0.5 of training set (Cora) 0.018 equals 0.5 (citeseer) 0.003 equals 0.5 (pubmed)
    parser.add_argument('--reduction_rate', type=float, default=0.03)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    print(args)
    path = "checkpoints/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_dataset(args.dataset)
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
                test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
                all_acc.append(test_acc)

            # val_loss_history.append(val_loss)
            if args.early_stopping > 0:
                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    print('ave_test_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

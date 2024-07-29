import time

import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from graphslim.condensation.utils import normalize_data, GCF
from graphslim.condensation.utils import sub_E, update_E
from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset import LargeDataLoader
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import StructureBasedNeuralTangentKernel, KernelRidgeRegression
from tqdm import trange
from graphslim.utils import seed_everything


class GCSNTK(GCondBase):
    """
    "GFast Graph Conensation with Structure-based Neural Tangent Kernel" https://arxiv.org/pdf/2310.11046
    """
    def __init__(self, setting, data, args, **kwargs):
        super(GCSNTK, self).__init__(setting, data, args, **kwargs)
        self.k = args.k
        self.K = args.K
        self.ridge = args.ridge
        self.L = args.L
        self.scale = args.scale

    def train(self, KRR, G_t, G_s, y_t, y_s, E_t, E_s, loss_fn, optimizer, accumulate_steps=None, i=None, TRAIN_K=None):
        pred, acc = KRR.forward(G_t, G_s, y_t, y_s, E_t, E_s)

        pred = pred.to(torch.float32)
        y_t = y_t.to(torch.float32)
        loss = loss_fn(pred, y_t)
        loss = loss.to(torch.float32)

        if accumulate_steps is None:
            # with torch.autograd.detect_anomaly():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = loss / accumulate_steps
            loss.backward()

            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            elif i == TRAIN_K - 1:
                optimizer.step()
                optimizer.zero_grad()

        loss = loss.item()

        # print(f"Training loss: {loss:>7f} Training Acc: {acc:>7f}", end=' ')
        return G_s, y_s, loss, acc * 100

    def test(self, KRR, G_t, G_s, y_t, y_s, E_t, E_s, loss_fn):
        size = len(y_t)
        test_loss, correct = 0, 0
        with torch.no_grad():
            pred, _ = KRR.forward(G_t, G_s, y_t, y_s, E_t, E_s)
            test_loss += loss_fn(pred, y_t).item()
            correct += (pred.argmax(1) == y_t.argmax(1)).type(torch.float).sum().item()
        correct /= size
        print(f"Val Acc: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}", end='\n')
        return test_loss, correct * 100

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        if args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
            train_loader = LargeDataLoader(name=self.args.dataset, split='train', batch_size=self.args.batch_size,
                                           split_method='kmeans')
            TRAIN_K, n_train, n_class, dim, n = train_loader.properties()
            train_loader.split_batch()
        else:
            edge_index = data.edge_index
            n_class = len(torch.unique(data.y))
            n, dim = data.x.shape

            adj = sp.coo_matrix((torch.ones(data.edge_index.shape[1]), edge_index), shape=(n, n)).toarray()
            adj = torch.tensor(adj)
            adj = adj + torch.eye(adj.shape[0])

            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
            y_train, y_val, y_test = data.labels_train, data.labels_val, data.labels_test
            y_one_hot = F.one_hot(data.y, n_class)
            y_train_one_hot = y_one_hot[data.train_mask]

            n_train = len(y_train)
        Cond_size = round(n_train * self.args.reduction_rate)
        # fixed initialization by gaussian
        x_s = torch.rand(round(Cond_size), dim, device=args.device)
        y_s = torch.rand(round(Cond_size), n_class, device=args.device)
        x_s.requires_grad = True
        y_s.requires_grad = True
        idx_s = torch.tensor(range(Cond_size))
        optimizer = torch.optim.Adam([x_s, y_s], lr=self.args.lr)

        SNTK = StructureBasedNeuralTangentKernel(K=self.K, L=self.L, scale=self.scale).to(self.device)
        ridge = torch.tensor(self.ridge).to(self.device)
        KRR = KernelRidgeRegression(SNTK.nodes_gram, ridge).to(self.device)
        MSEloss = nn.MSELoss().to(self.device)
        if args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:

            if self.args.adj:
                feat = x_s.data
                A_s = update_E(feat, 3)
            else:
                A_s = torch.sparse_coo_tensor(torch.stack([idx_s, idx_s], dim=0), torch.ones(Cond_size),
                                              torch.Size([Cond_size, Cond_size])).to(x_s.device)

            A_s = A_s.to(self.device)

            optimizer = torch.optim.Adam([x_s, y_s], lr=self.args.lr)

            best_val = 0
            for it in trange(args.epochs):
                for i in range(TRAIN_K):
                    x_train, label, sub_A_t = train_loader.get_batch(i)
                    y_train_one_hot = F.one_hot(label.reshape(-1), n_class)

                    x_train = x_train.to(self.device)
                    y_train_one_hot = y_train_one_hot.to(self.device)
                    sub_A_t = sub_A_t.to(self.device)

                    _, _, training_loss, train_correct = self.train(KRR, x_train, x_s, y_train_one_hot, y_s, sub_A_t,
                                                                    A_s,
                                                                    MSEloss, optimizer, args.accumulate_steps, i,
                                                                    TRAIN_K)
                if it in args.checkpoints:
                    # y_long = torch.argmax(y_s, dim=1)
                    data.adj_syn, data.feat_syn, data.labels_syn = A_s.detach().to_dense(), x_s.detach(), y_s.detach()
                    best_val = self.intermediate_evaluation(best_val, training_loss)


        else:
            E_train = sub_E(idx_train, adj).to(self.device)
            y_train_one_hot = y_train_one_hot.to(self.device)
            x_train = data.feat_train.to(self.device)

            if self.args.adj:
                feat = x_s.data
                A_s = update_E(feat, 4)
            else:
                A_s = torch.sparse_coo_tensor(torch.stack([idx_s, idx_s], dim=0), torch.ones(Cond_size),
                                              torch.Size([Cond_size, Cond_size])).to(x_s.device)

            best_val = 0
            for it in trange(args.epochs):

                x_s, y_s, training_loss, training_acc = self.train(KRR, x_train, x_s, y_train_one_hot, y_s, E_train,
                                                                   A_s, MSEloss, optimizer)

                if it in args.checkpoints:
                    # y_long = torch.argmax(y_s, dim=1)
                    data.adj_syn, data.feat_syn, data.labels_syn = A_s.detach().to_dense(), x_s.detach(), y_s.detach()
                    best_val = self.intermediate_evaluation(best_val, training_loss)

        return data

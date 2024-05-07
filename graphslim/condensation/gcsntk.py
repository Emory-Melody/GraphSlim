import time

import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from graphslim.condensation.utils import normalize_data, GCF
from graphslim.condensation.utils import sub_E, update_E
from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset import FlickrDataLoader
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import StructureBasedNeuralTangentKernel, KernelRidgeRegression
from tqdm import trange
from graphslim.utils import seed_everything


class GCSNTK(GCondBase):
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
        y_test_one_hot = y_one_hot[data.val_mask]
        # y_test_one_hot = y_one_hot[data.test_mask]
        x = normalize_data(data.x)
        x = GCF(adj, x, self.k)
        x_train, _, x_test = x[data.train_mask], x[data.val_mask], x[data.val_mask]

        n_train = len(y_train)
        Cond_size = round(n_train * self.args.reduction_rate)
        idx_s = torch.tensor(range(Cond_size))

        E_train = sub_E(idx_train, adj)
        E_test = sub_E(idx_val, adj)

        SNTK = StructureBasedNeuralTangentKernel(K=self.K, L=self.L, scale=self.scale).to(self.device)
        ridge = torch.tensor(self.ridge).to(self.device)
        KRR = KernelRidgeRegression(SNTK.nodes_gram, ridge).to(self.device)
        MSEloss = nn.MSELoss().to(self.device)

        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        E_test = E_test.to(self.device)
        E_train = E_train.to(self.device)

        y_train_one_hot = y_train_one_hot.to(self.device)
        y_test_one_hot = y_test_one_hot.to(self.device)

        x_s = torch.rand(Cond_size, dim).to(self.device)
        y_s = torch.rand(Cond_size, n_class).to(self.device)
        if self.args.adj:
            feat = x_s.data
            E_s = update_E(feat, 4)
        else:
            E_s = torch.sparse_coo_tensor(torch.stack([idx_s, idx_s], dim=0), torch.ones(Cond_size),
                                          torch.Size([Cond_size, Cond_size])).to(x_s.device)

        x_s.requires_grad = True
        y_s.requires_grad = True
        optimizer = torch.optim.Adam([x_s, y_s], lr=self.args.lr)

        best_val = 0
        for epoch in trange(args.epochs):
            x_s, y_s, training_loss, training_acc = self.train(KRR, x_train, x_s, y_train_one_hot, y_s, E_train,
                                                               E_s, MSEloss, optimizer)

            if epoch + 1 in args.checkpoints:
                # y_long = torch.argmax(y_s, dim=1)
                data.adj_syn, data.feat_syn, data.labels_syn = E_s.detach().to_dense(), x_s.detach(), y_s.detach()
                best_val = self.intermediate_evaluation(best_val, training_loss)
                # val_loss, val_acc = self.test(KRR, x_test, x_s, y_test_one_hot, y_s, E_test, E_s, MSEloss)
                # if val_acc > best_val:
                #     best_val = val_acc
                #     y_long = torch.argmax(y_s, dim=1)
                #     data.adj_syn, data.feat_syn, data.labels_syn = E_s.detach().to_dense(), x_s.detach(), y_long.detach()
                #     save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, self.args, best_val)

        return data

    @verbose_time_memory
    def reduce_huge(self, data, verbose=True):
        print("begin load")
        train_loader = FlickrDataLoader(name=self.args.dataset, split='train', batch_size=self.args.batch_size,
                                        split_method='kmeans')
        test_loader = FlickrDataLoader(name=self.args.dataset, split='val', batch_size=self.args.batch_size,
                                       split_method='kmeans')
        TRAIN_K, n_train, n_class, dim, n = train_loader.properties()
        test_k, n_test, _, _, _ = test_loader.properties()
        train_loader.split_batch()
        test_loader.split_batch()

        Cond_size = int(n_train * self.args.reduction_rate)
        idx_s = torch.tensor(range(round(Cond_size)))

        SNTK = StructureBasedNeuralTangentKernel(K=self.K, L=self.L, scale=self.scale).to(self.device)
        ridge = torch.tensor(self.ridge).to(self.device)
        kernel = SNTK.nodes_gram
        KRR = KernelRidgeRegression(kernel, ridge).to(self.device)

        results = torch.zeros(self.args.epochs_train, 1)
        for iter in range(1):
            print(f"The  {iter + 1}-th iteration")
            x_s = torch.rand(round(Cond_size), dim)
            y_s = torch.rand(round(Cond_size), n_class)
            if self.args.adj:
                feat = x_s.data
                A_s = update_E(feat, 3)
            else:
                A_s = torch.sparse_coo_tensor(torch.stack([idx_s, idx_s], dim=0), torch.ones(Cond_size),
                                              torch.Size([Cond_size, Cond_size])).to(x_s.device)

            MSEloss = nn.MSELoss().to(self.device)
            idx_s = idx_s.to(self.device)
            x_s = x_s.to(self.device)
            y_s = y_s.to(self.device)
            A_s = A_s.to(self.device)
            x_s.requires_grad = True
            y_s.requires_grad = True

            optimizer = torch.optim.Adam([x_s, y_s], lr=self.args.lr)

            max_test_acc = 0
            start = time.time()

            T = 0
            Time = []
            Time.append(T)

            for t in range(self.args.epochs_train):
                print(f"Epoch {t + 1}", end=" ")
                train_loss, test_lossi = torch.zeros(TRAIN_K), torch.zeros(test_k)
                train_correct_all, test_correct_all = 0, 0

                a = time.time()
                for i in range(TRAIN_K):
                    x_train, label, sub_A_t = train_loader.get_batch(i)
                    y_train_one_hot = F.one_hot(label.reshape(-1), n_class)

                    x_train = x_train.to(self.device)
                    y_train_one_hot = y_train_one_hot.to(self.device)
                    sub_A_t = sub_A_t.to(self.device)

                    _, _, training_loss, train_correct = self.train(KRR, x_train, x_s, y_train_one_hot, y_s, sub_A_t,
                                                                    A_s,
                                                                    MSEloss, optimizer, self.args.accumulate_steps, i,
                                                                    TRAIN_K)

                    train_correct_all = train_correct_all + train_correct
                    train_loss[i] = training_loss

                b = time.time()
                T = T + b - a
                Time.append(T)
                training_loss_avg = torch.mean(train_loss)
                training_acc_avg = (train_correct_all / n_train) * 100

                test_a = time.time()

                if t >= 1:
                    for j in range(test_k):
                        x_test, test_label, sub_A_test = test_loader.get_batch(j)
                        y_test_one_hot = F.one_hot(test_label.reshape(-1), n_class)

                        x_test = x_test.to(self.device)
                        y_test_one_hot = y_test_one_hot.to(self.device)
                        sub_A_test = sub_A_test.to(self.device)

                        test_loss, test_correct = self.test(KRR, x_test, x_s, y_test_one_hot, y_s, sub_A_test, A_s,
                                                            MSEloss)

                        test_correct_all = test_correct_all + test_correct
                        test_lossi[j] = test_loss

                    test_loss_avg = torch.mean(test_lossi)
                    test_acc = (test_correct_all / n_test) * 100
                    print(f"Test Acc: {(test_acc):>0.4f}%, Test loss: {test_loss_avg:>0.6f}", end='\n')
                    results[t, iter] = test_acc

                test_b = time.time()

            end = time.time()
            print('Running time: %s Seconds' % (end - start))

            print("---------------------------------------------")

        Acc_mean, Acc_std = torch.mean(results, dim=1), torch.std(results, dim=1)
        max_mean, max_mean_index = torch.max(Acc_mean, dim=0)
        print(f'Mean Test Acc: {max_mean.item():>0.4f}%, Std: {Acc_std[max_mean_index].item():>0.4f}%')
        print("--------------- Train Done! ----------------")

        y_s = torch.argmax(y_s, dim=1)
        data.adj_syn, data.feat_syn, data.labels_syn = A_s.detach().to_dense(), x_s.detach(), y_s.detach()

        save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, self.args)
        print("--------------- Save Done! ----------------")

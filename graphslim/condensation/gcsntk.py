import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from graphslim.condensation.utils import normalize_data, GCF
from graphslim.condensation.utils import sub_E, update_E
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import StructureBasedNeuralTangentKernel, KernelRidgeRegression
# from graphslim.dataset import load_data
from graphslim.utils import seed_everything


class GCSNTK:
    def __init__(self, setting, data, args, **kwargs):
        self.data = data
        self.args = args
        self.setting = setting
        self.device = args.device
        self.k = args.k
        self.K = args.K
        self.ridge = args.ridge
        self.L = args.L
        self.scale = args.scale

    def train(self, KRR, G_t, G_s, y_t, y_s, E_t, E_s, loss_fn, optimizer):
        pred, acc = KRR.forward(G_t, G_s, y_t, y_s, E_t, E_s)

        pred = pred.to(torch.float32)
        y_t = y_t.to(torch.float32)
        loss = loss_fn(pred, y_t)
        loss = loss.to(torch.float32)

        # with torch.autograd.detect_anomaly():
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()

        print(f"Training loss: {loss:>7f} Training Acc: {acc:>7f}", end=' ')
        return G_s, y_s, loss, acc * 100

    def test(self, KRR, G_t, G_s, y_t, y_s, E_t, E_s, loss_fn):
        size = len(y_t)
        test_loss, correct = 0, 0
        with torch.no_grad():
            pred, _ = KRR.forward(G_t, G_s, y_t, y_s, E_t, E_s)
            test_loss += loss_fn(pred, y_t).item()
            correct += (pred.argmax(1) == y_t.argmax(1)).type(torch.float).sum().item()
        correct /= size
        print(f"Test Acc: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}", end='\n')
        return test_loss, correct * 100

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        edge_index = data.edge_index
        n_class = len(torch.unique(data.y))
        n, dim = data.x.shape

        adj = sp.coo_matrix((torch.ones(data.edge_index.shape[1]), edge_index), shape=(n, n)).toarray()
        adj = torch.tensor(adj)
        adj = adj + torch.eye(adj.shape[0])

        idx_train, _, idx_test = data.idx_train, data.idx_val, data.idx_test
        y_train, _, y_test = data.labels_train, data.labels_val, data.labels_test
        y_one_hot = F.one_hot(data.y, n_class)
        y_train_one_hot = y_one_hot[data.train_mask]
        # y_val_one_hot = y_one_hot[data.val_mask]
        y_test_one_hot = y_one_hot[data.test_mask]
        x = normalize_data(data.x)
        x = GCF(adj, x, self.k)
        x_train, _, x_test = x[data.train_mask], x[data.val_mask], x[data.test_mask]

        n_train = len(y_train)
        Cond_size = round(n_train * self.args.reduction_rate)
        idx_s = torch.tensor(range(Cond_size))

        seed_everything(self.args.seed)

        E_train = sub_E(idx_train, adj)
        E_test = sub_E(idx_test, adj)

        SNTK = StructureBasedNeuralTangentKernel(K=self.K, L=self.L, scale=self.scale).to(self.device)
        ridge = torch.tensor(self.ridge).to(self.device)
        KRR = KernelRidgeRegression(SNTK.nodes_gram, ridge).to(self.device)
        MSEloss = nn.MSELoss().to(self.device)

        adj = adj.to(self.device)
        x = x.to(self.device)
        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        E_test = E_test.to(self.device)
        E_train = E_train.to(self.device)

        y_train_one_hot = y_train_one_hot.to(self.device)
        y_test_one_hot = y_test_one_hot.to(self.device)

        print(f"Dataset       :{self.args.dataset}")
        print(f"Training Set  :{len(y_train)}")
        print(f"Testing Set   :{len(y_test)}")
        print(f"Classes       :{n_class}")
        print(f"Dim           :{dim}")
        print(f"Number        :{n}")
        print(f"Epochs        :{self.args.epochs_train}")
        print(f"Learning rate :{self.args.lr}")
        print(f"Conden ratio  :{self.args.reduction_rate}")
        print(f"Ridge         :{self.ridge}")

        Acc = torch.zeros(self.args.epochs_train, self.args.iter).to(self.device)
        for iter in range(self.args.iter):
            print('--------------------------------------------------')
            print('The ' + str(iter + 1) + 'th Iteration:')
            print('--------------------------------------------------')

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

            for epoch in range(self.args.epochs_train):
                print(f"Epoch {epoch + 1}", end=" ")
                x_s, y_s, training_loss, training_acc = self.train(KRR, x_train, x_s, y_train_one_hot, y_s, E_train,
                                                                   E_s, MSEloss, optimizer)

                test_loss, test_acc = self.test(KRR, x_test, x_s, y_test_one_hot, y_s, E_test, E_s, MSEloss)
                Acc[epoch, iter] = test_acc

        Acc_mean, Acc_std = torch.mean(Acc, dim=1), torch.std(Acc, dim=1)

        print('Mean and std of test data : {:.4f}, {:.4f}'.format(Acc_mean[-1], Acc_std[-1]))
        print("--------------- Train Done! ----------------")

        y_s = torch.argmax(y_s, dim=1)
        data.adj_syn, data.feat_syn, data.labels_syn = E_s.detach().to_dense(), x_s.detach(), y_s.detach()

        save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, self.args)

        print("--------------- Save Done! ----------------")

        return data

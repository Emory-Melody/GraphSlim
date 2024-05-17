from torch import nn
from tqdm import trange

from collections import Counter
from sklearn.cluster import BisectingKMeans
from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import match_loss
from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from torch_scatter import scatter_mean
from graphslim.utils import *


class MSGC(GCondBase):

    def __init__(self, setting, data, args, **kwargs):
        super(MSGC, self).__init__(setting, data, args, **kwargs)
        x_channels = data.feat_train.shape[1]
        edge_hidden_channels = 256
        self.n_syn = self.nnodes_syn
        n_each_y = self.generate_labels_syn(data)
        self.labels_syn = data.labels_syn = self.y_syn
        self.feat_syn = nn.Parameter(torch.empty(self.n_syn, x_channels).to(args.device))
        self.batch_size = args.batch_adj
        self.n_classes = data.nclass
        self.device = args.device

        self.num_class_dict = self.data.num_class_dict = {index: value for index, value in enumerate(n_each_y)}

        self.adj_mlp = nn.Sequential(
            nn.Linear(x_channels * 2, edge_hidden_channels),
            nn.BatchNorm1d(edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, edge_hidden_channels),
            nn.BatchNorm1d(edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, 1)
        ).to(args.device)
        # -------------------------------------------------------------------------
        # self.reset_adj_batch()

    def generate_labels_syn(self, data):
        labels_train = data.labels_train.to(self.device)
        n = labels_train.shape[0]
        n_syn = self.nnodes_syn
        base = torch.ones(data.nclass, device=self.device)
        rate = F.one_hot(labels_train, num_classes=data.nclass).sum(0) / n
        n_each_y = torch.floor((n_syn - base.sum()) * rate) + base
        left = n_syn - n_each_y.sum()
        for _ in range(int(left.item())):
            more = n_each_y / n_each_y.sum() / rate
            n_each_y[more.argmin()] += 1
        n_each_y = n_each_y.to(torch.int64)

        y_syn = torch.LongTensor(n_syn).to(self.device)
        start = 0
        starts = torch.zeros_like(n_each_y)
        for c in range(data.nclass):
            y_syn[start:start + n_each_y[c]] = c
            starts[c] = start
            start += n_each_y[c]
        self.y_syn = y_syn
        if self.args.verbose:
            print(f'num_class:{n_each_y}')
        return n_each_y

    def reduce(self, data, verbose=True):

        args = self.args
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        adj = normalize_adj_tensor(adj, sparse=True)
        y_syn = to_tensor(label=self.y_syn, device=self.device).repeat(self.batch_size)
        # assert args.condense_model != 'GAT'
        basic_model = eval(args.condense_model)(self.feat_syn.shape[1], args.hidden, data.nclass, args).to(self.device)

        self.reset_adj_batch()
        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)

        optimizer_x = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        optimizer_adj = torch.optim.Adam(self.adj_mlp.parameters(), lr=args.lr_adj)
        best_val = 0
        args.window = args.patience = 20
        losses = FixLenList(args.window)
        x_syns = FixLenList(args.window)
        adj_t_syns = FixLenList(args.window)
        optimizer_basic_model = torch.optim.Adam(basic_model.parameters(), lr=args.lr)

        for it in trange(args.epochs):
            basic_model.initialize()
            basic_model = self.check_bn(basic_model)
            loss_avg = 0
            for step_syn in range(args.outer_loop):
                basic_model = self.check_bn(basic_model)
                basic_model.eval()  # fix basic_model while optimizing graphsyner
                ######################graph optimization#####################################
                self.adj_syn = self.get_adj_t_syn()

                loss = self.train_class(basic_model, adj, features, labels, y_syn, args)
                loss_avg += loss.item()
                optimizer_x.zero_grad()
                optimizer_adj.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    optimizer_adj.step()
                else:
                    optimizer_x.step()

                x_syn = self.feat_syn.detach()
                adj_t_syn = self.get_adj_t_syn().detach()
                #################################################
                for i in range(args.inner_loop):
                    optimizer_basic_model.zero_grad()
                    logits = basic_model(x_syn, adj_t_syn)
                    inner_loss = F.nll_loss(logits, y_syn)
                    inner_loss.backward()
                    optimizer_basic_model.step()
            loss_avg /= (data.nclass * args.outer_loop)
            losses.append(loss_avg)
            x_syns.append(x_syn.clone())
            adj_t_syns.append(adj_t_syn.clone())
            loss_window = sum(losses.data) / len(losses.data)
            if args.verbose:
                print(f'loss:{loss_window:.4f} feat:{x_syn.sum().item():.4f} adj:{adj_t_syn.sum().item():.4f}')
            if it in args.checkpoints:
                best_x_syn = sum(x_syns.data) / len(x_syns.data)
                best_adj_t_syn = sum(adj_t_syns.data) / len(adj_t_syns.data)
                data.feat_syn, data.adj_syn, data.labels_syn = best_x_syn, best_adj_t_syn, y_syn
                best_val = self.intermediate_evaluation(best_val, loss_window)
            # if loss_window < smallest_loss:
            #     patience = 0
            #     smallest_loss = loss_window
            #     best_x_syn = sum(x_syns.data) / len(x_syns.data)
            #     best_adj_t_syn = sum(adj_t_syns.data) / len(adj_t_syns.data)
            #     # print(f'loss:{smallest_loss:.4f} feat:{x_syn.sum().item():.4f} adj:{adj_t_syn.sum().item():.4f}')
            # else:
            #     patience += 1
            #     if patience >= args.patience:
            #         break
        # save according to loss
        # data.feat_syn, data.adj_syn, data.labels_syn = best_x_syn, best_adj_t_syn, y_syn
        # best_val = self.intermediate_evaluation(0, loss_window)

        return data

    # def init(self, with_adj=False):
    #     n_classes = self.data.nclass
    #     y_syn = self.y_syn
    #     # cluster is restricted in training set in MSGC.
    #     x_train = self.data.feat_train
    #     y_train = self.data.labels_train
    #     if self.init == 'cluster':
    #         x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1])
    #         for c in range(n_classes):
    #             x_c = x_train[y_train == c].cpu()
    #             n_c = (y_syn == c).sum().item()
    #             k_means = BisectingKMeans(n_clusters=n_c, random_state=0)
    #             k_means.fit(x_c)
    #             clusters = torch.LongTensor(k_means.predict(x_c))
    #             x_syn[y_syn == c] = scatter_mean(x_c, clusters, dim=0)
    #         return x_syn.to(x_train.device)
    #     elif self.init == 'sample':
    #         sam = SamplerForClass(y_train, n_classes)
    #         counter = Counter(y_syn.cpu().numpy())
    #         idx_selected_list = []
    #         for c in range(n_classes):
    #             idx_c = sam.sample_from_class(c, n_need_max=counter[c])
    #             idx_selected_list.append(idx_c)
    #         idx_selected = torch.cat(idx_selected_list).to(x_train.device)
    #         return x_train[idx_selected]
    #     elif self.init == 'mean':
    #         x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1]).to(x_train.device)
    #         for c in range(n_classes):
    #             x_c = x_train[y_train == c]
    #             n_c = (y_syn == c).sum()
    #             x_syn[y_syn == c] = x_c.mean(0)
    #         return x_syn

    def reset_adj_batch(self):
        rows = []
        cols = []
        batch = []
        for i in range(self.batch_size):
            n_neighbor = torch.zeros(self.n_syn, self.n_classes, device=self.device)
            index = torch.arange(self.n_syn, device=self.device)
            row = []
            col = []
            for row_id in range(self.n_syn):
                for c in range(self.n_classes):
                    c_mask = self.y_syn == c
                    c_mask[row_id] = False
                    if c_mask.sum() == 0 or n_neighbor[row_id][c] > 1:
                        continue
                    link_coef = n_neighbor[c_mask, self.y_syn[row_id]]
                    selected = link_coef.argmin()
                    candidates_mask = link_coef == link_coef[selected]
                    if candidates_mask.sum() == 1:
                        col_id = index[c_mask][selected].item()
                    else:
                        candidates = index[c_mask][candidates_mask]
                        col_id = candidates[torch.randperm(candidates.shape[0], device=self.device)[0]].item()
                    n_neighbor[row_id, c] += 1
                    n_neighbor[col_id, self.y_syn[row_id]] += 1
                    row.append(row_id)
                    row.append(col_id)
                    col.append(col_id)
                    col.append(row_id)
            rows.append(torch.LongTensor(row))
            cols.append(torch.LongTensor(col))
            batch.append(torch.LongTensor([i] * len(row)))

        self.rows = torch.cat(rows).to(self.device)
        self.cols = torch.cat(cols).to(self.device)
        self.batch = torch.cat(batch).to(self.device)
        n_edge = self.rows.shape[0] / self.batch_size / 2
        if self.args.verbose:
            print(f'n_edge:{n_edge}')

    def get_adj_t_syn(self):
        adj = torch.zeros(size=(self.batch_size, self.n_syn, self.n_syn), device=self.device)
        adj[self.batch, self.rows, self.cols] = torch.sigmoid(self.adj_mlp(
            torch.cat([self.feat_syn[self.rows], self.feat_syn[self.cols]], dim=1)).flatten())
        adj = (torch.transpose(adj, 1, 2) + adj) / 2

        adj += torch.eye(self.n_syn, device=self.device).view(1, self.n_syn, self.n_syn)
        deg = adj.sum(2)
        deg_inv = deg.pow(-1 / 2)
        deg_inv[torch.isinf(deg_inv)] = 0.
        adj = adj * deg_inv.view(self.batch_size, -1, 1)
        adj = adj * deg_inv.view(self.batch_size, 1, -1)
        return adj


class FixLenList:
    def __init__(self, lenth):
        self.lenth = lenth
        self.data = []

    def append(self, element):
        self.data.append(element)
        if len(self.data) > self.lenth:
            del self.data[0]

from tqdm import trange
import copy
import os.path as osp

import networkx as nx
from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import *
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *
from torch.nn import Parameter
from torch import nn


class GDEM(GCondBase):
    """
    "Graph Condensation for Graph Neural Networks" https://cse.msu.edu/~jinwei2/files/GCond.pdf
    """

    def __init__(self, setting, data, args, **kwargs):
        super(GDEM, self).__init__(setting, data, args, **kwargs)
        args.eigen_k = min(args.eigen_k, self.nnodes_syn)
        self.eigenvecs_syn = Parameter(
            torch.FloatTensor(self.nnodes_syn, args.eigen_k).to(self.device)
        )

        self.y_syn = to_tensor(label=self.labels_syn, device=self.device)
        self.x_syn = self.feat_syn

        init_syn_feat = self.init_feat()
        init_syn_eigenvecs = self.get_init_syn_eigenvecs(self.nnodes_syn, data.nclass)
        init_syn_eigenvecs = init_syn_eigenvecs[:, :args.eigen_k]
        self.x_syn.data.copy_(init_syn_feat)
        self.eigenvecs_syn.data.copy_(init_syn_eigenvecs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
        # calculate eigenvalues and eigenvectors
        dataset_dir = osp.join(args.load_path, args.dataset)
        if not osp.exists(f"{dataset_dir}/idx_map.npy"):
            idx_lcc, adj_norm_lcc, _ = get_largest_cc(data.adj_full, data.num_nodes, args.dataset)
            np.save(f"{dataset_dir}/idx_lcc.npy", idx_lcc)

            L_lcc = sp.eye(len(idx_lcc)) - adj_norm_lcc
            get_eigh(L_lcc, f"{args.dataset}", True)

            idx_train_lcc, idx_map = get_train_lcc(idx_lcc=idx_lcc, idx_train=data.idx_train, y_full=data.y,
                                                   num_nodes=data.num_nodes, num_classes=data.nclass)
            np.save(f"{dataset_dir}/idx_train_lcc.npy", idx_train_lcc)
            np.save(f"{dataset_dir}/idx_map.npy", idx_map)
            print('preprocess done')
        else:
            idx_lcc = np.load(f"{dataset_dir}/idx_lcc.npy")
            idx_train_lcc = np.load(f"{dataset_dir}/idx_train_lcc.npy")
            idx_map = np.load(f"{dataset_dir}/idx_map.npy")

        eigenvals_lcc, eigenvecs_lcc = load_eigen(args.dataset,args.load_path)
        eigenvals_lcc = torch.FloatTensor(eigenvals_lcc)
        eigenvecs_lcc = torch.FloatTensor(eigenvecs_lcc)

        n_syn = int(len(data.idx_train) * args.reduction_rate)
        eigenvals, eigenvecs = get_syn_eigen(real_eigenvals=eigenvals_lcc, real_eigenvecs=eigenvecs_lcc,
                                             eigen_k=args.eigen_k,
                                             ratio=args.ratio)

        co_x_trans_real = get_subspace_covariance_matrix(eigenvecs, data.feat_full[idx_lcc]).to(self.device)
        embed_sum = get_embed_sum(eigenvals=eigenvals, eigenvecs=eigenvecs, x=data.feat_full[idx_lcc])
        embed_sum = embed_sum[idx_map, :]
        embed_mean_real = get_embed_mean(embed_sum=embed_sum, label=data.y[idx_train_lcc]).to(self.device)
        optimizer_feat = self.optimizer_feat
        optimizer_eigenvec = torch.optim.Adam(
            [self.eigenvecs_syn], lr=args.lr_eigenvec
        )
        eigenvals_syn = eigenvals.to(self.device)
        best_val = 0
        bar = trange(args.epochs)
        for ep in bar:
            loss = 0.0
            x_syn = self.x_syn
            eigenvecs_syn = self.eigenvecs_syn

            # eigenbasis match
            co_x_trans_syn = get_subspace_covariance_matrix(eigenvecs=eigenvecs_syn, x=x_syn).to(self.device)  # kdd
            eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
            loss += args.alpha * eigen_match_loss

            # class loss
            embed_sum_syn = get_embed_sum(eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=x_syn).to(self.device)
            embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=self.y_syn).to(self.device)  # cd
            cov_embed = embed_mean_real @ embed_mean_syn.T
            iden = torch.eye(data.nclass, device=self.device)
            class_loss = F.mse_loss(cov_embed, iden)
            loss += args.beta * class_loss

            # orthog_norm
            orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
            iden = torch.eye(args.eigen_k, device=self.device)
            orthog_norm = F.mse_loss(orthog_syn, iden)
            loss += args.gamma * orthog_norm
            if verbose and ep in args.checkpoints:
                print(f"epoch: {ep}")
                print(f"eigen_match_loss: {eigen_match_loss}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * eigen_match_loss}")

                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")

                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")

            optimizer_eigenvec.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()

            # update U:
            if ep % (args.e1 + args.e2) < args.e1:
                optimizer_eigenvec.step()
            else:
                optimizer_feat.step()

            eigenvecs_syn = self.eigenvecs_syn.detach()
            eigenvals_syn = eigenvals_syn.detach()

            if ep in args.checkpoints:
                L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T
                adj_syn = torch.eye(self.nnodes_syn, device=self.device) - L_syn
                x_syn, y_syn = self.x_syn.detach(), self.y_syn
                data.adj_syn, data.feat_syn, data.labels_syn = adj_syn, x_syn, y_syn
                best_val = self.intermediate_evaluation(best_val)

        return data

    def get_eigenspace_embed(self, eigen_vecs, x):
        eigen_vecs = eigen_vecs.unsqueeze(2)  # k * n * 1
        eigen_vecs_t = eigen_vecs.permute(0, 2, 1)  # k * 1 * n
        eigenspace = torch.bmm(eigen_vecs, eigen_vecs_t)  # knn
        embed = torch.matmul(eigenspace, x)  # knn*nd=knd
        return embed

    def get_real_embed(self, k, L, x):
        filtered_x = x

        emb_list = []
        for i in range(k):
            filtered_x = L @ filtered_x
            emb_list.append(filtered_x)

        embed = torch.stack(emb_list, dim=0)
        return embed

    def get_syn_embed(self, k, eigenvals, eigen_vecs, x):
        trans_x = eigen_vecs @ x
        filtered_x = trans_x

        emb_list = []
        for i in range(k):
            filtered_x = torch.diag(eigenvals) @ filtered_x
            emb_list.append(eigen_vecs.T @ filtered_x)

        embed = torch.stack(emb_list, dim=0)
        return embed

    def get_init_syn_feat(self, dataset, reduction_rate):
        expID = self.args.seed
        init_syn_x = torch.load(f"./initial_feat/{dataset}/x_init_{reduction_rate}_{expID}.pt", map_location="cpu")
        return init_syn_x

    def get_init_syn_eigenvecs(self, n_syn, num_classes):
        n_nodes_per_class = n_syn // num_classes
        n_nodes_last = n_syn % num_classes

        size = [n_nodes_per_class for i in range(num_classes - 1)] + (
            [n_syn - (num_classes - 1) * n_nodes_per_class] if n_nodes_last != 0 else [n_nodes_per_class]
        )
        prob_same_community = 1 / num_classes
        prob_diff_community = prob_same_community / 3

        prob = [
            [prob_diff_community for i in range(num_classes)]
            for i in range(num_classes)
        ]
        for idx in range(num_classes):
            prob[idx][idx] = prob_same_community

        syn_graph = nx.stochastic_block_model(size, prob)
        syn_graph_adj = nx.adjacency_matrix(syn_graph)
        syn_graph_L = gcn_normalize_adj(syn_graph_adj)
        syn_graph_L = np.eye(n_syn) - syn_graph_L
        _, eigen_vecs = get_eigh(syn_graph_L, "", False)

        return torch.FloatTensor(eigen_vecs).to(self.device)

    def mlp_trainer(self, args, data, verbose):
        x_full, y_full = to_tensor(data.feat_full, label=data.labels_full, device=self.device)

        model = MLP(num_features=x_full.shape[1], num_classes=data.nclass, hidden_dim=args.hidden,
                    dropout=args.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc_val = 0
        y_train, y_val, y_test = to_tensor(label1=data.labels_train, label2=data.labels_val, label=data.labels_test,
                                           device=self.device)
        feat_train, feat_val = to_tensor(data.feat_train, data.feat_val, device=self.device)

        lr = args.lr
        for i in range(2000):
            if i == 1000 // 2 and i > 0:
                lr = lr * 0.1
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )

            model.train()
            optimizer.zero_grad()

            output = model.forward(feat_train)
            loss_train = F.nll_loss(output, y_train)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output = model.forward(feat_val)
                # loss_val = F.nll_loss(output, y_val)

                acc_val = accuracy(output, y_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    weights = copy.deepcopy(model.state_dict())

        model.load_state_dict(weights)

        return model

    def init_feat(self):
        args = self.args
        data = self.data
        labels_syn = to_tensor(label=data.labels_syn, device=self.device)

        optimizer_feat = torch.optim.Adam(
            [self.x_syn], lr=args.lr_feat, weight_decay=0
        )

        model = self.mlp_trainer(args, data, verbose=False)
        model.train()

        for i in range(2000):
            output = model(self.x_syn)
            loss = F.nll_loss(output, labels_syn)
            #
            optimizer_feat.zero_grad()
            loss.backward()
            optimizer_feat.step()
        return self.x_syn


class MLP(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            hidden_dim,
            dropout):

        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_dim), nn.Linear(hidden_dim, num_classes)])
        self.reset_parameter()

    def reset_parameter(self):
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight.data)
            if lin.bias is not None:
                lin.bias.data.zero_()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

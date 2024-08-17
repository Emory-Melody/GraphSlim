from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *
from torch_sparse import SparseTensor


class SimGC(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(SimGC, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
        device = self.device
        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=device)
        self.reset_parameters()
        # feat_syn.data.copy_(self.init())
        # feat_syn.data.copy_(torch.randn_like(feat_syn.data))
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train, device=device)
        # student_model = eval(args.condense_model)(self.d, args.hidden, data.nclass, args).to(device)

        if args.dataset in ["cora", "citeseer"]:
            args.ntrans = 2
            args.lr = args.lr_teacher
            teacher_model = SGC(nfeat=self.d, nhid=args.hidden, nclass=data.nclass, args=args).to(device)
            teacher_val_acc = teacher_model.fit_with_val(data, train_iters=10000, verbose=args.verbose, normadj=False,
                                                         setting=args.setting,
                                                         reduced=False)
        else:
            args.dropout = 0.5
            args.ntrans = 2
            args.nlayers = 3
            args.with_bn = True
            teacher_model = SGC(nfeat=self.d, nhid=args.hidden, nclass=data.nclass, args=args).to(device)
            teacher_val_acc = teacher_model.fit_with_val(data, train_iters=1000, verbose=args.verbose, normadj=False,
                                                         setting=args.setting,
                                                         reduced=False)
            # recover to default value
            args.with_bn = False
            args.lr = 0.01
            args.hidden = 256
            args.dropout = 0
            args.nlayers = 2
        args.logger.info(f"Teacher model training finished with val accuracy {teacher_val_acc}")

        optimizer_feat, optimizer_pge = self.optimizer_feat, self.optimizer_pge
        adj = normalize_adj_tensor(adj, sparse=True)

        # alignment
        concat_feat = data.feat_train.to(self.device)
        temp = features

        for i in range(args.nlayers):
            aggr = (adj @ temp).detach()
            concat_feat = torch.cat((concat_feat, aggr[data.idx_train]), dim=1)
            temp = aggr

        concat_feat_mean = []
        concat_feat_std = []
        coeff = []
        coeff_sum = 0
        # init the mean and std of each class
        for c in range(data.nclass):
            index = torch.where(data.labels_train == c)
            coe = self.num_class_dict[c] / max(self.num_class_dict.values())
            coeff_sum += coe
            coeff.append(coe)
            concat_feat_mean.append(concat_feat[index].mean(dim=0).to(device))
            concat_feat_std.append(concat_feat[index].std(dim=0).to(device))

        n = feat_syn.shape[0]
        best_val = 0
        for it in trange(args.epochs + 1):
            teacher_model.eval()
            optimizer_pge.zero_grad()
            optimizer_feat.zero_grad()

            adj_syn = pge(feat_syn).to(device)
            adj_syn[adj_syn < args.threshold] = 0
            edge_index_syn = torch.nonzero(adj_syn).T
            edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]

            # smoothness loss
            feat_difference = torch.exp(-0.5 * torch.pow(feat_syn[edge_index_syn[0]] - feat_syn[edge_index_syn[1]], 2))
            smoothness_loss = torch.dot(edge_weight_syn, torch.mean(feat_difference, 1).flatten()) / torch.sum(
                edge_weight_syn)

            edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, n)
            sparse_adj_syn = SparseTensor(row=edge_index_syn[0], col=edge_index_syn[1], value=edge_weight_syn,
                                          sparse_sizes=(n, n))
            concat_feat_syn = feat_syn.to(device)
            temp = feat_syn
            for j in range(args.nlayers):
                aggr_syn = (sparse_adj_syn @ temp).detach()
                concat_feat_syn = torch.cat((concat_feat_syn, aggr_syn), dim=1)
                temp = aggr_syn
            # inversion loss

            output_syn = teacher_model.forward(feat_syn, sparse_adj_syn)
            hard_loss = F.nll_loss(output_syn, labels_syn)

            # alignment loss
            concat_feat_loss = torch.tensor(0.0).to(device)
            loss_fn = torch.nn.MSELoss()
            for c in range(data.nclass):
                index = torch.where(labels_syn == c)
                concat_feat_mean_loss = coeff[c] * loss_fn(concat_feat_mean[c], concat_feat_syn[index].mean(dim=0))
                concat_feat_std_loss = coeff[c] * loss_fn(concat_feat_std[c], concat_feat_syn[index].std(dim=0))
                if feat_syn[index].shape[0] != 1:
                    concat_feat_loss += (concat_feat_mean_loss + concat_feat_std_loss)
                else:
                    concat_feat_loss += (concat_feat_mean_loss)
            concat_feat_loss = concat_feat_loss / coeff_sum

            # total loss
            loss = hard_loss + args.feat_alpha * concat_feat_loss + args.smoothness_alpha * smoothness_loss
            loss.backward()
            if it % 50 < 10:
                optimizer_pge.step()
            else:
                optimizer_feat.step()

            if it in args.checkpoints:
                adj_syn = pge(feat_syn).detach().to(device)
                adj_syn[adj_syn < args.threshold] = 0
                adj_syn.requires_grad = False
                self.adj_syn = adj_syn
                output_syn = teacher_model.predict(feat_syn, adj_syn)
                acc = accuracy(output_syn, labels_syn).item()
                if verbose:
                    print(f"Train Accuracy {acc:.4f}")

                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss)

        return data

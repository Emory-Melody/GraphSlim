from copy import copy
from collections import Counter
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor
from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import SGC
from graphslim.utils import accuracy, normalize_adj_tensor, to_tensor


def training_view(features, data, setting):
    """Global indices apply only to the full transductive graph."""
    return features[data.idx_train] if setting == "trans" else features


def propagated_features(features, adj, depth):
    """Keep synthetic multi-hop representations differentiable (author code)."""
    hops = [features]
    for _ in range(depth):
        hops.append(adj @ hops[-1])
    return torch.cat(hops, dim=1)


def smoothness_loss(features, edge_index, edge_weight):
    """Minimize negative similarity, as in the released SimGC implementation."""
    similarity = torch.exp(-0.5 * (features[edge_index[0]] - features[edge_index[1]]).pow(2))
    return -(edge_weight * similarity.mean(dim=1)).sum() / edge_weight.sum().clamp_min(1e-12)


class SimGCTeacher(SGC):
    """Author SGC: propagate first, then apply the linear/nonlinear predictor."""

    def __init__(self, nfeat, nhid, nclass, args, mode="train"):
        super().__init__(nfeat, nhid, nclass, args, mode)
        initializer = None if self.ntrans > 1 else "glorot"
        self.layers = torch.nn.ModuleList(
            Linear(layer.in_features, layer.out_features, weight_initializer=initializer)
            for layer in self.layers
        )

    def forward(self, x, adj, output_layer_features=False):
        for hop in range(self.nlayers):
            x = (adj[hop] if isinstance(adj, list) else adj) @ x
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index != len(self.layers) - 1:
                x = self.bns[index](x) if self.with_bn else x
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        output = F.log_softmax(x, dim=1)
        return (x, output) if output_layer_features else output


class SimGC(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(SimGC, self).__init__(setting, data, args, **kwargs)

    def generate_labels_syn(self, data):
        """Use the released SimGC per-class ceiling, not the generic budget floor."""
        labels = []
        self.syn_class_indices, self.num_class_dict = {}, {}
        counter = Counter(data.labels_train.tolist())
        for label, count in sorted(counter.items(), key=lambda pair: pair[1]):
            size = ceil(count * self.args.reduction_rate)
            self.num_class_dict[label] = size
            self.syn_class_indices[label] = [len(labels), len(labels) + size]
            labels.extend([label] * size)
        data.num_class_dict = self.num_class_dict
        return np.asarray(labels, dtype=np.int64)

    def validation_model_args(self):
        args = copy(self.args)
        explicit = getattr(args, "_explicit_cli_keys", set())
        for key, value in getattr(args, "validation_config", {}).items():
            if key not in explicit:
                setattr(args, key, value)
        return args

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
        device = self.device
        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=device)
        self.reset_parameters()
        if args.setting == "trans":
            features, adj = to_tensor(data.feat_full, data.adj_full, device=device)
        else:
            features, adj = to_tensor(data.feat_train, data.adj_train, device=device)

        teacher_args = copy(args)
        citation = args.dataset in ["cora", "citeseer"]
        teacher_args.ntrans = 2 if citation else 1
        teacher_args.nlayers = 2
        teacher_args.with_bn = False
        teacher_args.dropout = 0
        teacher_args.activation = "relu"
        teacher_args.val_interval = getattr(args, "teacher_val_interval", 1)
        teacher_args.weight_decay = 0
        teacher_args.lr = getattr(args, "lr_teacher", 0.001 if args.setting == "trans" else 0.01)
        teacher_args.lr_decay = False
        teacher_model = SimGCTeacher(
            nfeat=self.d,
            nhid=getattr(args, "teacher_hidden", 2048 if citation else args.hidden),
            nclass=data.nclass,
            args=teacher_args,
        ).to(device)
        teacher_val_acc = teacher_model.fit_with_val(
            data,
            train_iters=getattr(args, "teacher_epochs", 1000 if args.setting == "trans" else 600),
            verbose=args.verbose,
            normadj=True,
            setting=args.setting,
            reduced=False,
        )
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        args.logger.info(f"Teacher model training finished with val accuracy {teacher_val_acc}")

        optimizer_feat, optimizer_pge = self.optimizer_feat, self.optimizer_pge
        adj = normalize_adj_tensor(adj, sparse=True)

        # alignment
        concat_feat = data.feat_train.to(self.device)
        temp = features

        for i in range(2):
            aggr = (adj @ temp).detach()
            concat_feat = torch.cat((concat_feat, training_view(aggr, data, args.setting)), dim=1)
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

            smoothness = smoothness_loss(feat_syn, edge_index_syn, edge_weight_syn)

            edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, n)
            sparse_adj_syn = SparseTensor(
                row=edge_index_syn[0],
                col=edge_index_syn[1],
                value=edge_weight_syn,
                sparse_sizes=(n, n),
            )
            concat_feat_syn = propagated_features(feat_syn, sparse_adj_syn, 2)
            # inversion loss

            output_syn = teacher_model.forward(feat_syn, sparse_adj_syn)
            hard_loss = F.nll_loss(output_syn, labels_syn)

            # alignment loss
            concat_feat_loss = torch.tensor(0.0).to(device)
            loss_fn = torch.nn.MSELoss()
            for c in range(data.nclass):
                index = torch.where(labels_syn == c)
                concat_feat_mean_loss = coeff[c] * loss_fn(
                    concat_feat_mean[c], concat_feat_syn[index].mean(dim=0)
                )
                if feat_syn[index].shape[0] != 1:
                    concat_feat_std_loss = coeff[c] * loss_fn(
                        concat_feat_std[c], concat_feat_syn[index].std(dim=0)
                    )
                    concat_feat_loss += concat_feat_mean_loss + concat_feat_std_loss
                else:
                    concat_feat_loss += concat_feat_mean_loss
            concat_feat_loss = concat_feat_loss / coeff_sum

            # total loss
            loss = (
                hard_loss + args.feat_alpha * concat_feat_loss + args.smoothness_alpha * smoothness
            )
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

                data.adj_syn, data.feat_syn, data.labels_syn = (
                    self.adj_syn.detach(),
                    self.feat_syn.detach(),
                    labels_syn.detach(),
                )
                best_val = self.intermediate_evaluation(best_val, loss)

        return data

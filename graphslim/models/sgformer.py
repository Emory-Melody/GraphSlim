# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from graphslim.utils import *
from graphslim.dataset import *
from torch import optim
from copy import deepcopy
from torch.nn import NLLLoss


def normalize_adj(adj):
    """Normalize dense adjacency by weighted degree or sparse adjacency by edge counts.

    :param adj: Dense Torch matrix or coalesced native sparse COO matrix. The sparse
        path uses column edge counts for its degree factors.
    :returns: Normalized adjacency in the input layout, with non-finite normalized
        entries replaced by zero. No self-loops are added.
    """
    # Extract indices and values from the sparse COO tensor
    if is_sparse_tensor(adj):
        row, col = adj.indices()
        values = adj.values()

        # Number of nodes
        N = adj.size(0)

        # Compute degree for normalization
        d = degree(col, N).float()
        d_norm_in = (1.0 / d[col]).sqrt()
        d_norm_out = (1.0 / d[row]).sqrt()

        # Normalize the values directly
        normalized_values = values * d_norm_in * d_norm_out
        normalized_values = torch.nan_to_num(normalized_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Create a new sparse COO tensor with normalized values
        adj_normalized = torch.sparse_coo_tensor(
            torch.stack([row, col]), normalized_values, size=(N, N)
        )
    else:
        N = adj.size(0)

        # Compute degree for normalization
        d = adj.sum(dim=1).float()
        d_norm = torch.diag((1.0 / d).sqrt())

        # Normalize the dense adjacency matrix
        adj_normalized = d_norm @ adj @ d_norm
        adj_normalized = torch.nan_to_num(adj_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return adj_normalized
    return adj_normalized


class GraphConvLayer(nn.Module):
    """SGFormer graph propagation layer with optional initial-feature concatenation."""

    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        """Initialize GraphConvLayer and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param out_channels: Output representation width.
        :param use_weight: Whether to learn a linear feature transformation. Defaults to ``True``.
        :param use_init: Whether to include the initial node representation in each layer. Defaults to ``False``.
        """
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        self.W.reset_parameters()

    def forward(self, x, adj, x0):
        # N = x.shape[0]
        # row, col = edge_index
        # d = degree(col, N).float()
        # d_norm_in = (1. / d[col]).sqrt()
        # d_norm_out = (1. / d[row]).sqrt()
        # value = torch.ones_like(row) * d_norm_in * d_norm_out
        # value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        # adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        """Normalize graph adjacency, propagate features, and optionally concatenate initial representations.

        :param x: Node feature matrix with one row per node.
        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param x0: Initial node representations used by layers configured with ``use_init``.
        :returns: Updated node representations.
        """
        adj = normalize_adj(adj)
        x = adj @ x

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    """SGFormer graph branch with configurable normalization and residual connections."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_init=False,
        use_act=True,
    ):
        """Initialize GraphConv and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param hidden_channels: Hidden representation width.
        :param num_layers: Number of layers. Defaults to ``2``.
        :param dropout: Dropout probability. Defaults to ``0.5``.
        :param use_bn: Whether to use batch normalization. Defaults to ``True``.
        :param use_residual: Whether to add residual connections. Defaults to ``True``.
        :param use_weight: Whether to learn a linear feature transformation. Defaults to ``True``.
        :param use_init: Whether to include the initial node representation in each layer. Defaults to ``False``.
        :param use_act: Whether to apply nonlinear activation between layers. Defaults to ``True``.
        """
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        """Run the graph branch with its configured normalization, dropout, and residual options.

        :param x: Node feature matrix with one row per node.
        :param edge_index: Integer edge endpoints with shape ``(2, num_edges)``.
        :returns: Graph branch node representations.
        """
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class TransConvLayer(nn.Module):
    """
    transformer with fast attention
    """

    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        """Initialize TransConvLayer and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param out_channels: Output representation width.
        :param num_heads: Number of attention heads.
        :param use_weight: Whether to learn a linear feature transformation. Defaults to ``True``.
        """
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        """Compute normalized multihead linear attention between query and source nodes.

        :param query_input: Query or key/value node feature matrix, respectively.
        :param source_input: Query or key/value node feature matrix, respectively.
        :param output_attn: Whether to also return the dense node attention matrix. Defaults to ``False``.
        :returns: Attention output, or ``(output, attention_matrix)`` when attention weights are requested.
        """
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        # Identical constants; avoid a pageable CPU->GPU transfer inside capture.
        all_ones = torch.ones([ks.shape[0]], device=ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    """SGFormer transformer branch built from linear attention layers."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_act=True,
    ):
        """Initialize TransConv and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param hidden_channels: Hidden representation width.
        :param num_layers: Number of layers. Defaults to ``2``.
        :param num_heads: Number of attention heads. Defaults to ``1``.
        :param dropout: Dropout probability. Defaults to ``0.5``.
        :param use_bn: Whether to use batch normalization. Defaults to ``True``.
        :param use_residual: Whether to add residual connections. Defaults to ``True``.
        :param use_weight: Whether to learn a linear feature transformation. Defaults to ``True``.
        :param use_act: Whether to apply nonlinear activation between layers. Defaults to ``True``.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(
                    hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        """Run stacked all-pairs linear attention layers with configured residual and normalization options.

        :param x: Node feature matrix with one row per node.
        :returns: Transformer branch node representations.
        """
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.0
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        """Collect attention weights from each transformer layer.

        :param x: Node feature matrix with one row per node.
        :returns: Tensor of shape ``(num_layers, num_nodes, num_nodes)``.
        """
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.0
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    """Node classifier combining global linear attention and a graph convolution branch."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        args,
        mode="eval",
        trans_num_layers=3,
        trans_num_heads=1,
        trans_dropout=0.2,
        trans_weight_decay=0.001,
        trans_use_bn=False,
        trans_use_residual=True,
        trans_use_weight=False,
        trans_use_act=False,
        gnn_num_layers=2,
        gnn_dropout=0.5,
        gnn_use_weight=False,
        gnn_weight_decay=5e-4,
        gnn_use_init=False,
        gnn_use_bn=False,
        gnn_use_residual=False,
        gnn_use_act=False,
        use_graph=True,
        graph_weight=0.8,
        aggregate="add",
    ):
        """Initialize SGFormer and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param hidden_channels: Hidden representation width.
        :param out_channels: Output representation width.
        :param args: Configuration namespace containing the options used by this operation.
        :param mode: Model configuration mode; selects training or evaluation settings. Defaults to ``'eval'``.
        :param trans_num_layers: Number of layers in this branch. Defaults to ``3``.
        :param trans_num_heads: Number of attention heads in the transformer branch. Defaults to ``1``.
        :param trans_dropout: Dropout probability in this branch. Defaults to ``0.2``.
        :param trans_weight_decay: Optimizer weight decay applied to this branch. Defaults to ``0.001``.
        :param trans_use_bn: Whether this branch uses batch normalization. Defaults to ``False``.
        :param trans_use_residual: Whether this branch uses residual connections. Defaults to ``True``.
        :param trans_use_weight: Whether this branch learns a feature transform during propagation. Defaults to ``False``.
        :param trans_use_act: Whether this branch applies nonlinear activations. Defaults to ``False``.
        :param gnn_num_layers: Number of layers in this branch. Defaults to ``2``.
        :param gnn_dropout: Dropout probability in this branch. Defaults to ``0.5``.
        :param gnn_use_weight: Whether this branch learns a feature transform during propagation. Defaults to ``False``.
        :param gnn_weight_decay: Optimizer weight decay applied to this branch. Defaults to ``0.0005``.
        :param gnn_use_init: Whether graph layers concatenate the initial node representation. Defaults to ``False``.
        :param gnn_use_bn: Whether this branch uses batch normalization. Defaults to ``False``.
        :param gnn_use_residual: Whether this branch uses residual connections. Defaults to ``False``.
        :param gnn_use_act: Whether this branch applies nonlinear activations. Defaults to ``False``.
        :param use_graph: Whether to include the graph convolution branch. Defaults to ``True``.
        :param graph_weight: Weight of the graph branch when branch outputs are added. Defaults to ``0.8``.
        :param aggregate: Branch combination: ``add`` for weighted addition, otherwise concatenation. Defaults to ``'add'``.
        """
        super().__init__()
        self.args = args
        self.device = args.device
        self.lr = args.lr
        if hasattr(args, "trans_weight_decay"):
            self.trans_weight_decay = args.trans_weight_decay
        else:
            self.trans_weight_decay = trans_weight_decay
        if hasattr(args, "trans_dropout"):
            self.trans_dropout = args.trans_dropout
        else:
            self.trans_dropout = trans_dropout
        if hasattr(args, "trans_num_layers"):
            self.trans_num_layers = args.trans_num_layers
        else:
            self.trans_num_layers = trans_num_layers
        self.gnn_weight_decay = gnn_weight_decay
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            self.trans_num_layers,
            trans_num_heads,
            self.trans_dropout,
            trans_use_bn,
            trans_use_residual,
            trans_use_weight,
            trans_use_act,
        )
        self.graph_conv = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "cat":
            # Without the graph branch only the transformer output is concatenated.
            self.fc = nn.Linear((2 if use_graph else 1) * hidden_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        """Combine transformer and optional graph branches and classify each node.

        :param x: Node feature matrix with one row per node.
        :param edge_index: Integer edge endpoints with shape ``(2, num_edges)``.
        :returns: Class log probabilities of shape ``(num_nodes, num_classes)``.
        """
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_attentions(self, x):
        """Return attention matrices from the transformer branch.

        :param x: Node feature matrix with one row per node.
        :returns: Tensor of shape ``(num_transformer_layers, num_nodes, num_nodes)``.
        """
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()

    def fit_with_val(
        self,
        data,
        train_iters=600,
        verbose=False,
        normadj=True,
        setting="trans",
        reduced=False,
        reindex=False,
        **kwargs,
    ):
        """Train SGFormer and restore the state with the highest validation metric.

        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param train_iters: Number of training iterations. Defaults to ``600``.
        :param verbose: Whether to print progress and diagnostic information. Defaults to ``False``.
        :param normadj: Whether to normalize the adjacency before use. Defaults to ``True``.
        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs. Defaults to ``'trans'``.
        :param reduced: Whether to use the synthetic graph instead of the original training graph. Defaults to ``False``.
        :param reindex: Whether to reindex reduced-graph training outputs through the original training indices. Defaults to ``False``.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        :returns: Best validation metric score.
        """
        args = self.args

        self.reset_parameters()
        # data for training
        if reduced:
            adj, features, labels, labels_val = to_tensor(
                data.adj_syn,
                data.feat_syn,
                data.labels_syn,
                label2=data.labels_val,
                device=self.device,
            )
        elif setting == "trans":
            adj, features, labels, labels_val = to_tensor(
                data.adj_full,
                data.feat_full,
                label=data.labels_train,
                label2=data.labels_val,
                device=self.device,
            )
        else:
            adj, features, labels, labels_val = to_tensor(
                data.adj_train,
                data.feat_train,
                label=data.labels_train,
                label2=data.labels_val,
                device=self.device,
            )

        # edge_index = torch.stack([adj.indices()[0], adj.indices()[1]], dim=0).to(self.device)

        if labels.ndim == 2:
            labels = labels.float()
            if args.eval_loss == "MSE":
                self.loss = lambda output, target: F.mse_loss(output.exp(), target)
            else:
                self.loss = lambda output, target: -(target.softmax(1) * output).sum(1).mean()
        else:
            labels = to_tensor(label=labels, device=self.device)
            self.loss = F.nll_loss

        # elif len(data.labels_full.shape) > 1:
        #     self.multi_label = True
        #     self.loss = torch.nn.BCELoss()
        # else:
        #     self.multi_label = False
        #     self.loss = F.nll_loss

        if reduced or setting == "ind":
            reindex = True

        if verbose:
            print("=== training ===")

        best_acc_val = 0
        weights = None
        if setting == "ind":
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = to_tensor(feat_full, adj_full, device=self.device)
        optimizer = torch.optim.Adam(
            [
                {"params": self.params1, "weight_decay": self.trans_weight_decay},
                {"params": self.params2, "weight_decay": self.gnn_weight_decay},
            ],
            lr=self.lr,
        )

        training_index = data.idx_train

        def training_loss():
            """Compute SGFormer predictions and the current training objective.

            :returns: Tuple ``(node_predictions, scalar_training_loss)``.
            """
            output = self.forward(features, adj)
            return output, self.loss(output if reindex else output[training_index], labels)

        self.train()
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, loss_train = training_loss()
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print("Epoch {}, training loss: {}".format(i, loss_train.item()))
                hard_labels = labels.argmax(1) if labels.ndim == 2 else labels
                acc_train = accuracy(output if reindex else output[data.idx_train], hard_labels)
                print("Epoch {}, training acc: {}".format(i, acc_train))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full)

                if setting == "ind":
                    # loss_val = F.nll_loss(output, labels_val)
                    acc_val = args.metric(output, labels_val)
                else:
                    # loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = args.metric(output[data.idx_val], labels_val)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    # self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print("=== picking the best model according to the performance on validation ===")
        # Keep the final weights when validation never improves, as BaseGNN does.
        if weights is not None:
            self.load_state_dict(weights)
        return best_acc_val

    @torch.no_grad()
    def test(self, data, setting="trans", verbose=False):
        """Evaluate the selected SGFormer on the test graph or full-graph test indices.

        :param data: Graph data containing features, adjacency, labels, and split indices.
        :param setting: Graph learning setting: ``trans`` for the full graph or ``ind`` for split subgraphs. Defaults to ``'trans'``.
        :param verbose: Whether to print progress and diagnostic information. Defaults to ``False``.
        :returns: Configured test metric as a Python float in the interval [0, 1].
        """
        self.eval()
        idx_test = data.idx_test
        labels_test = torch.LongTensor(data.labels_test).to(self.device)
        # whether condensed or not, use the raw graph to test

        if setting == "ind":
            output = self.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = self.args.metric(output, labels_test)
        else:
            output = self.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[idx_test], labels_test)
            acc_test = self.args.metric(output[idx_test], labels_test)

        if verbose:
            print(
                "Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                f"{getattr(self.args.metric, '__name__', 'score')}= {float(acc_test):.4f}",
            )
        return float(acc_test)

    @torch.no_grad()
    def predict(self, features=None, adj=None, normadj=True, output_layer_features=False):
        """Convert graph inputs to device tensors and run evaluation-mode prediction.

        ``normadj`` and ``output_layer_features`` are accepted for the shared model
        interface; this method directly returns class predictions.

        :param features: Node feature matrix with one row per node. Defaults to ``None``.
        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features. Defaults to ``None``.
        :param normadj: Whether to normalize the adjacency before use. Defaults to ``True``.
        :param output_layer_features: Whether to also return intermediate layer features with the predictions. Defaults to ``False``.
        :returns: Class log probability tensor.
        """
        self.eval()
        features, adj = to_tensor(features, adj, device=self.device)
        return self.forward(features, adj)

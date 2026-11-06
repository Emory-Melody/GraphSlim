# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
# License: MIT. PyG and pygcn adaptations; see ../THIRD_PARTY_NOTICES.md.
import math
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptPairTensor, Adj, Size, NoneType, OptTensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor, set_diag, matmul
from torch_geometric.utils import dense_to_sparse


class GraphConvolution(torch.nn.Module):
    """Learnable graph convolution for dense, batched, or torch_sparse adjacency."""

    def __init__(self, in_features, out_features, with_bias=True, weight_initializer=None):
        """Initialize GraphConvolution and its configuration.

        :param in_features: Number of input features per node.
        :param out_features: Output representation width.
        :param with_bias: Whether to include a learnable additive bias. Defaults to ``True``.
        :param weight_initializer: Optional name selecting the linear weight initialization rule. Defaults to ``None``.
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_initializer = weight_initializer
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.zeros(out_features))  # change this line
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        if self.weight_initializer == "glorot":
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
            return
        stdv = 1.0 / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """Transform node features and aggregate them through dense, batched, or sparse adjacency.

        :param x: Node feature matrix with one row per node.
        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :returns: Graph-convolved node features including optional bias.
        """
        if isinstance(adj, SparseTensor):
            x = torch.mm(x, self.weight)
            x = adj @ x
        else:
            x = x.view(-1, x.shape[-1])
            x = torch.mm(x, self.weight)
            x = x.view(-1, adj.shape[-1], x.shape[-1])
            x = adj @ x
        x.view(-1, x.shape[-1])
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def __repr__(self):
        """Describe the layer dimensions and configuration.

        :returns: Human-readable layer representation.
        """
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GATConv(MessagePassing):
    r"""
    The graph attentional operator from
    `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`_ [1]_:

        .. math::
            \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
            \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j}

        where the attention coefficients :math:`\alpha_{i,j}` are computed as:

        .. math::
            \alpha_{i,j} =
            \frac{
            \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
            [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
            \right)\right)}
            {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
            \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
            [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
            \right)\right)}

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments for
            :class:`torch_geometric.nn.conv.MessagePassing`.

    References
    ----------
    .. [1] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1710.10903
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize GATConv and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param out_channels: Output representation width.
        :param heads: Number of attention heads. Defaults to ``1``.
        :param concat: Whether to concatenate head outputs instead of averaging them. Defaults to ``True``.
        :param negative_slope: Negative slope used by the attention LeakyReLU activation. Defaults to ``0.2``.
        :param dropout: Dropout probability. Defaults to ``0.0``.
        :param add_self_loops: Whether to add diagonal edges during adjacency preparation. Defaults to ``True``.
        :param bias: Linear classifier bias vector. Defaults to ``True``.
        :param kwargs: Additional keyword options passed to the underlying implementation.
        """
        kwargs.setdefault("aggr", "add")
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()
        self.edge_weight = None

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        return_attention_weights=None,
        edge_weight=None,
    ):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        """Apply multi-head graph attention and optionally expose attention weights.

        :param x: Node feature matrix or a pair of source and target feature
            matrices for a bipartite graph; the target matrix may be ``None``.
        :param edge_index: Integer edge endpoints or a transposed
            ``torch_sparse.SparseTensor`` adjacency.
        :param size: Optional ``(num_source_nodes, num_target_nodes)`` pair used
            for message passing and self-loop insertion. Defaults to ``None``.
        :param return_attention_weights: ``None`` returns only node features.
            Either boolean value returns attention weights as well, following
            the type-based return convention. Defaults to ``None``.
        :param edge_weight: Optional message weights for the tensor edge-index
            path with self-loop insertion enabled. Weights must align with the
            resulting edge list and are cached on the layer. Defaults to ``None``.
        :returns: Updated node features, optionally paired with
            ``(edge_index, attention_weights)`` or a weighted sparse adjacency.
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

                if edge_weight is not None:
                    if self.edge_weight is None:
                        self.edge_weight = edge_weight

                    if edge_index.size(1) != self.edge_weight.shape[0]:
                        self.edge_weight = edge_weight

            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        """Normalize edge attention coefficients and form weighted neighbor messages.

        :param x_j: Source node features lifted to each message edge.
        :param alpha_j: Source and target attention logits per edge, respectively.
        :param alpha_i: Source and target attention logits per edge, respectively.
        :param index: Target node index for each message, used by segmented attention softmax.
        :param ptr: Optional compressed sparse row pointers for segmented softmax.
        :param size_i: Number of target nodes used by message aggregation.
        :returns: Head-wise neighbor messages after attention dropout.
        """
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # from .gpu_memory_log import gpu_memory_log
        if self.edge_weight is not None:
            x_j = self.edge_weight.view(-1, 1, 1) * x_j
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        """Describe the layer dimensions and configuration.

        :returns: Human-readable layer representation.
        """
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class SageConvolution(torch.nn.Module):
    """GraphSAGE layer combining self and neighbor transformations."""

    def __init__(self, in_channels, out_channels):
        """Initialize SageConvolution and its configuration.

        :param in_channels: Input representation width, or a pair of source and target widths where supported.
        :param out_channels: Output representation width.
        """
        super(SageConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        self.lin_r.reset_parameters()

    def forward(self, x, adj_t):
        """Add transformed self features and adjacency-aggregated neighbor features.

        :param x: Node feature matrix with one row per node.
        :param adj_t: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :returns: Updated node feature tensor; adjacency normalization is supplied by the caller.
        """
        if isinstance(adj_t, SparseTensor):
            h = matmul(adj_t, x, reduce="add")
            out = self.lin_r(h)
            out += self.lin_r(x[: adj_t.size(0)])
        else:
            h1 = self.lin_r(x)
            h1 = h1.view(-1, adj_t.shape[-1], h1.shape[-1])
            h2 = adj_t @ x.view(-1, adj_t.shape[-1], x.shape[-1])
            h2 = h2.view(-1, h2.shape[-1])
            h2 = self.lin_r(h2)
            h2 = h2.view(-1, adj_t.shape[-1], h2.shape[-1])
            out = h1 + h2
        out.view(-1, out.shape[-1])
        return out

    def __repr__(self):
        """Describe the layer dimensions and configuration.

        :returns: Human-readable layer representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class ChebConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn"""

    def __init__(self, in_features, out_features, with_bias=True, single_param=True, K=2):
        """Initialize a Chebyshev convolution with shared or per-order feature weights.

        :param in_features: Number of input features per node.
        :param out_features: Output representation width.
        :param with_bias: Whether to include a learnable additive bias. Defaults to ``True``.
        :param single_param: Whether polynomial orders share one feature transformation to limit parameter count. Defaults to ``True``.
        :param K: Number of Chebyshev terms in the convolution. Defaults to ``2``.
        """
        super(ChebConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lins = torch.nn.ModuleList(
            [MyLinear(in_features, out_features, with_bias=False) for _ in range(K)]
        )
        # self.lins = torch.nn.ModuleList([
        #    MyLinear(in_features, out_features, with_bias=True) for _ in range(K)])
        if with_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.single_param = single_param
        self.reset_parameters()

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, adj, size=None):
        """Apply the Chebyshev polynomial feature recurrence to graph adjacency.

        :param x: Node feature matrix with one row per node.
        :param adj: Graph adjacency; rows and columns refer to the same node ordering as the features.
        :param size: Size of the output or requested budget. Defaults to ``None``.
        :returns: Updated node feature tensor.
        """
        # support = torch.mm(input, self.weight_l)
        Tx_0 = x[: size[1]] if size is not None else x
        Tx_1 = x  # dummy
        output = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            if isinstance(adj, SparseTensor):
                Tx_1 = matmul(adj, x)
            else:
                Tx_1 = adj @ x

            if self.single_param:
                output = output + self.lins[0](Tx_1)
            else:
                output = output + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            if self.single_param:
                lin = self.lins[0]
            if isinstance(adj, SparseTensor):
                Tx_2 = matmul(adj, Tx_1)
            else:
                Tx_2 = adj @ Tx_1
            Tx_2 = 2.0 * Tx_2 - Tx_0
            output = output + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """Describe the layer dimensions and configuration.

        :returns: Human-readable layer representation.
        """
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MyLinear(torch.nn.Module):
    """Simple Linear layer, modified from https://github.com/tkipf/pygcn"""

    def __init__(self, in_features, out_features, with_bias=True):
        """Initialize MyLinear and its configuration.

        :param in_features: Number of input features per node.
        :param out_features: Output representation width.
        :param with_bias: Whether to include a learnable additive bias. Defaults to ``True``.
        """
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset trainable layers and normalization state using their initialization rules."""
        stdv = 1.0 / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Apply a learnable affine transformation to the last feature dimension.

        :param input: Input feature tensor.
        :returns: Transformed tensor with ``out_features`` entries in its last dimension.
        """
        support = input @ self.weight
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """Describe the layer dimensions and configuration.

        :returns: Human-readable layer representation.
        """
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

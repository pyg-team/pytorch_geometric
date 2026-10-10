from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    add_remaining_self_loops,
    coalesce,
    is_sparse,
    is_torch_sparse_tensor,
    scatter,
    softmax,
    to_edge_index,
)


def egnn_norm(edge_index: Tensor, edge_attr: Tensor,
              num_nodes: int) -> Tuple[Tensor, Tensor]:
    r"""Applies the doubly stochastic normalization from equations (1)-(2)
    of the EGNN paper, returning the resulting graph in COO format.

    Rows correspond to target nodes. Empty rows in an edge channel are given
    a unit self-loop before normalization. Unlike row normalization, this
    operation can introduce edges between nodes sharing a source neighbor.
    """
    dtype = edge_attr.dtype
    if dtype in [torch.float16, torch.bfloat16]:
        edge_attr = edge_attr.float()
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    # Scaling rows first avoids overflow/underflow in their sums.
    row_max = scatter(edge_attr.detach(), edge_index[1], dim=0,
                      dim_size=num_nodes, reduce='max')
    row_max = row_max.masked_fill(row_max == 0, 1.)
    edge_attr = edge_attr / row_max[edge_index[1]]
    row_sum = scatter(edge_attr, edge_index[1], dim=0, dim_size=num_nodes)

    # Complete empty rows independently for every edge channel.
    loop_attr = (row_sum == 0).to(edge_attr.dtype)
    nodes = torch.arange(num_nodes, device=edge_index.device)
    edge_index = torch.cat([edge_index, torch.stack([nodes, nodes])], dim=1)
    edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    row_sum = row_sum + loop_attr
    edge_attr = edge_attr / row_sum[edge_index[1]]
    col_sum = scatter(edge_attr, edge_index[0], dim=0, dim_size=num_nodes)

    indices, values = [], []
    for p in range(edge_attr.size(-1)):
        # DS(E) = T diag(T^T 1)^(-1) T^T, where T is row-normalized E.
        # Sparse products retain the new edges without an N x N allocation.
        value = edge_attr[:, p]
        keep = value != 0
        index = edge_index[:, keep].flip(0)
        value = value[keep]
        adj = torch.sparse_coo_tensor(index, value, (num_nodes, num_nodes))
        scaled = torch.sparse_coo_tensor(index, value / col_sum[index[1], p],
                                         (num_nodes, num_nodes))
        out = torch.sparse.mm(scaled, adj.transpose(0, 1)).coalesce()
        indices.append(out.indices().flip(0))
        values.append(out.values())

    edge_index = coalesce(torch.cat(indices, dim=1), num_nodes=num_nodes)
    keys = edge_index[0] * num_nodes + edge_index[1]
    # Align each channel with the union of its resulting edges without
    # expanding every intermediate sparse value into a P-dimensional vector.
    edge_attr = torch.stack([
        scatter(value, torch.searchsorted(
            keys, index[0] * num_nodes + index[1]), dim=0,
                dim_size=keys.numel())
        for index, value in zip(indices, values)
    ], dim=-1)
    return edge_index, edge_attr.to(dtype)


class EGNNConv(MessagePassing):
    r"""The edge-enhanced graph neural network operator from the
    `"Exploiting Edge Features in Graph Neural Networks"
    <https://arxiv.org/abs/1809.02709>`_ paper.

    Each of the :math:`P` edge channels guides a separate aggregation, using
    a shared node transformation :math:`\mathbf{\Theta}`:

    .. math::
        \mathbf{x}'_i = \mathop{\Vert}_{p=1}^P
        \sum_j E'_{ijp}\mathbf{\Theta}\mathbf{x}_j.

    With :obj:`backbone="gcn"` (EGNN(C)), :math:`\mathbf{E}'=\mathbf{E}`.
    With :obj:`backbone="gat"` (EGNN(A)), the edges evolve according to

    .. math::
        E'_{ijp} = \operatorname{DS}\left(
        E_{ijp}\exp\left(\operatorname{LeakyReLU}\left(
        \mathbf{a}_t^\top\mathbf{\Theta}\mathbf{x}_i +
        \mathbf{a}_s^\top\mathbf{\Theta}\mathbf{x}_j\right)\right)\right),

    where :math:`\operatorname{DS}` denotes the paper's exact doubly
    stochastic normalization, rather than iterative Sinkhorn normalization.
    Its output is symmetric and may contain edges absent from the input.
    The same attention function is shared by all edge channels.

    .. note::
        This is the edge-enhanced EGNN of Gong and Cheng, rather than the
        E(n)-equivariant graph neural network.

        Sparse inputs follow the :pyg:`PyG` convention of transposed adjacency
        matrices, with edge features stored as their values. Returned edges
        always use COO indices in source-to-target order. Exact normalization
        can produce a dense graph even from sparse input.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to infer
            it from the first input.
        out_channels (int): Output dimensionality **per edge channel**.
        edge_dim (int): Number of edge channels.
        backbone (str, optional): :obj:`"gcn"` or :obj:`"gat"`.
            (default: :obj:`"gat"`)
        normalize (bool, optional): Whether to normalize input edges. Disable
            when passing already normalized edges from a preceding layer.
            Attention weights are always doubly stochastic normalized.
            (default: :obj:`True`)
        add_self_loops (bool, optional): Whether to add missing self-loops
            with value :obj:`1` before input normalization. Existing loop
            values are preserved. Empty rows in each channel are completed
            with unit self-loops whenever normalization is performed.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU slope.
            (default: :obj:`0.2`)
        dropout (float, optional): Dropout on normalized attention weights
            used for aggregation. Undropped weights evolve to the next layer.
            Only used with :obj:`backbone="gat"`. (default: :obj:`0.0`)
        bias (bool, optional): Whether to learn an additive bias.
            (default: :obj:`True`)

    Shapes:
        - **input:** node features :math:`(N, F_{in})`, edge indices
          :math:`(2, M)`, nonnegative finite edge features :math:`(M, P)`
          (or :math:`(M,)` for a single channel). Omitted edge features default
          to ones in all channels.
        - **output:** node features :math:`(N, P F_{out})`; optionally, evolved
          edge indices :math:`(2, M')` and edge features :math:`(M', P)`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        backbone: str = 'gat',
        normalize: bool = True,
        add_self_loops: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)
        if backbone not in ['gcn', 'gat']:
            raise ValueError("'backbone' must be 'gcn' or 'gat'")
        if edge_dim < 1 or out_channels < 1:
            raise ValueError("'edge_dim' and 'out_channels' must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("'dropout' must be between 0 and 1")
        if add_self_loops and not normalize:
            raise ValueError("Adding self-loops requires input normalization")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.backbone = backbone
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        if backbone == 'gat':
            self.att_src = Parameter(torch.empty(1, out_channels))
            self.att_dst = Parameter(torch.empty(1, out_channels))
        else:
            self.register_parameter('att_src', None)
            self.register_parameter('att_dst', None)
        if bias:
            self.bias = Parameter(torch.empty(edge_dim * out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_edge_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        r"""Computes node features and optionally returns the evolved graph.

        Set :obj:`return_edge_weights=True` to pass the resulting edge indices
        and undropped edge weights to another :class:`EGNNConv` layer.
        """
        if x.dim() != 2:
            raise ValueError("'x' must be a two-dimensional tensor")
        num_nodes = x.size(0)
        if is_sparse(edge_index):
            if edge_attr is not None:
                raise ValueError("Sparse inputs store 'edge_attr' as values")
            if (edge_index.size(0) != num_nodes
                    or edge_index.size(1) != num_nodes):
                raise ValueError("Expected a square adjacency matching 'x'")
            if (is_torch_sparse_tensor(edge_index)
                    and edge_index.layout == torch.sparse_coo):
                adj = edge_index.coalesce()
                edge_index, edge_attr = adj.indices(), adj.values()
            else:
                edge_index, edge_attr = to_edge_index(edge_index)
            edge_index = edge_index.flip(0)
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), self.edge_dim))
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.shape != (edge_index.size(1), self.edge_dim):
            raise ValueError("Expected 'edge_attr' with shape [num_edges, "
                             "edge_dim]")
        if not torch.isfinite(edge_attr).all() or (edge_attr < 0).any():
            raise ValueError("Edge features must be finite and nonnegative")
        edge_attr = edge_attr.to(x.dtype)

        if self.add_self_loops:
            edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
            edge_index, edge_attr = add_remaining_self_loops(
                edge_index, edge_attr, fill_value=1., num_nodes=num_nodes)
        if self.normalize:
            edge_index, edge_attr = egnn_norm(edge_index, edge_attr, num_nodes)

        x = self.lin(x)
        if self.backbone == 'gat':
            alpha_src = (x * self.att_src).sum(-1)
            alpha_dst = (x * self.att_dst).sum(-1)
            alpha = alpha_src[edge_index[0]] + alpha_dst[edge_index[1]]
            alpha = F.leaky_relu(alpha, self.negative_slope)
            # log(E) + logits computes exp(logits) * E without overflow.
            # Replace zeros only for the logarithm; they remain absent below.
            logits = alpha.unsqueeze(-1) + edge_attr.masked_fill(
                edge_attr == 0, 1.).log()
            logits = logits.masked_fill(edge_attr == 0,
                                        torch.finfo(edge_attr.dtype).min)
            weights = softmax(logits, edge_index[1], num_nodes=num_nodes)
            weights = weights.masked_fill(edge_attr == 0, 0.)
            edge_index, edge_attr = egnn_norm(edge_index, weights, num_nodes)

        weights = edge_attr
        if self.backbone == 'gat':
            weights = F.dropout(weights, self.dropout, self.training)
        # propagate_type: (x: Tensor, edge_weight: Tensor)
        out = self.propagate(edge_index, x=x, edge_weight=weights,
                             size=(num_nodes, num_nodes))
        out = out.reshape(num_nodes, self.edge_dim * self.out_channels)
        if self.bias is not None:
            out = out + self.bias
        if return_edge_weights:
            return out, (edge_index, edge_attr)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.unsqueeze(-1) * x_j.unsqueeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, edge_dim={self.edge_dim}, '
                f"backbone='{self.backbone}')")

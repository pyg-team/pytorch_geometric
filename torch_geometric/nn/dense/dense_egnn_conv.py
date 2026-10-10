from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros


class DenseEGNNConv(torch.nn.Module):
    r"""The dense version of :class:`~torch_geometric.nn.conv.EGNNConv`.

    Accepts an adjacency tensor with shape :math:`(B, N, N, P)` or
    :math:`(N, N, P)`, where rows are target nodes and columns are source
    nodes. A single-channel adjacency still requires the final dimension
    :math:`P=1`. Node features and adjacency tensors are broadcastable in
    the batch dimension. Optional masks exclude padded nodes from both
    normalization and aggregation.

    See :class:`~torch_geometric.nn.conv.EGNNConv` for constructor arguments.
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
        super().__init__()
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
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
        return_edge_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Computes node features and optionally the evolved dense adjacency.

        Args:
            x (torch.Tensor): Node features with shape :math:`(B, N, F)`
                or :math:`(N, F)`.
            adj (torch.Tensor): Nonnegative finite edge features with shape
                :math:`(B, N, N, P)` or :math:`(N, N, P)`.
            mask (torch.Tensor, optional): Boolean mask with shape
                :math:`(B, N)` or :math:`(N,)`. (default: :obj:`None`)
            return_edge_weights (bool, optional): Whether to return the
                undropped evolved adjacency alongside the node features.
                (default: :obj:`False`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 3 else adj
        if x.dim() != 3 or adj.dim() != 4:
            raise ValueError("Expected 'x' with shape [B, N, F] and 'adj' "
                             "with shape [B, N, N, edge_dim]")
        num_nodes = x.size(1)
        if adj.shape[1:] != (num_nodes, num_nodes, self.edge_dim):
            raise ValueError("Expected a square adjacency matching 'x' "
                             "and 'edge_dim'")
        if not torch.isfinite(adj).all() or (adj < 0).any():
            raise ValueError("Edge features must be finite and nonnegative")
        adj = adj.to(x.dtype)
        batch_size = max(x.size(0), adj.size(0))
        x = x.expand(batch_size, -1, -1)
        adj = adj.expand(batch_size, -1, -1, -1)
        if mask is None:
            mask = torch.ones((batch_size, num_nodes), dtype=torch.bool,
                              device=x.device)
        else:
            mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
            if (mask.dim() != 2 or mask.size(1) != num_nodes
                    or mask.dtype != torch.bool):
                raise ValueError("Expected a boolean 'mask' with shape [B, N]")
            mask = mask.expand(batch_size, -1)
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        adj = adj * valid.unsqueeze(-1)
        nodes = torch.arange(num_nodes, device=x.device)
        if self.add_self_loops:
            adj = adj.clone()
            loop = adj[:, nodes, nodes]
            missing = (loop == 0).all(dim=-1) & mask
            adj[:, nodes, nodes] = torch.where(missing.unsqueeze(-1), 1., loop)
        if self.normalize:
            adj = self._normalize_adj(adj, mask)

        x = self.lin(x)
        if self.backbone == 'gat':
            alpha_src = (x * self.att_src).sum(-1)
            alpha_dst = (x * self.att_dst).sum(-1)
            logits = F.leaky_relu(
                alpha_dst.unsqueeze(2) + alpha_src.unsqueeze(1),
                self.negative_slope)
            logits = logits.unsqueeze(-1) + adj.masked_fill(adj == 0, 1.).log()
            logits = logits.masked_fill(adj == 0, torch.finfo(adj.dtype).min)
            weights = logits.softmax(dim=2).masked_fill(adj == 0, 0.)
            adj = self._normalize_adj(weights, mask)

        weights = adj
        if self.backbone == 'gat':
            weights = F.dropout(weights, self.dropout, self.training)
        out = weights.movedim(-1, 1) @ x.unsqueeze(1)
        out = out.movedim(1, 2).reshape(batch_size, num_nodes,
                                        self.edge_dim * self.out_channels)
        if self.bias is not None:
            out = out + self.bias
        out = out * mask.unsqueeze(-1)
        if return_edge_weights:
            return out, adj
        return out

    @staticmethod
    def _normalize_adj(adj: Tensor, mask: Tensor) -> Tensor:
        r"""Applies equations (1)-(2) independently to each edge channel."""
        nodes = torch.arange(adj.size(1), device=adj.device)
        loop = ((adj.sum(dim=2) == 0) & mask.unsqueeze(-1)).to(adj.dtype)
        adj = adj.clone()
        adj[:, nodes, nodes] = adj[:, nodes, nodes] + loop
        if adj.size(1) > 0:
            row_max = adj.detach().amax(dim=2, keepdim=True)
            adj = adj / row_max.masked_fill(row_max == 0, 1.)
        row_sum = adj.sum(dim=2, keepdim=True)
        adj = adj / row_sum.masked_fill(row_sum == 0, 1.)
        col_sum = adj.sum(dim=1, keepdim=True)
        scaled = adj / col_sum.masked_fill(col_sum == 0, 1.)
        return (scaled.movedim(-1, 1) @ adj.movedim(-1, 1).transpose(
            -1, -2)).movedim(1, -1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, edge_dim={self.edge_dim}, '
                f"backbone='{self.backbone}')")

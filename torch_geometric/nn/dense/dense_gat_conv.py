from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear

from ..inits import glorot, zeros


class DenseGATConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GATConv`.
    """

    # We don't consider edge attributes
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # In dense case, we no longer consider the case of bipartite graphs:
        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, 1, heads,
                                              out_channels))  # [1, 1, H, C]
        self.att_dst = Parameter(torch.Tensor(1, 1, heads,
                                              out_channels))  # [1, 1, H, C]

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None,
                add_loop: bool = True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \{0,1\}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        # first, x = theta * x through a linear model
        out = self.lin(x).view(B, N, H, C)  # [B, N, H, C]
        alpha_src = torch.sum(out * self.att_src, dim=-1)  # [B, N, H]
        alpha_dst = torch.sum(out * self.att_dst, dim=-1)  # [B, N, H]
        alpha = alpha_src.unsqueeze(2).repeat(
            1, 1, N, 1) + alpha_dst.unsqueeze(1).repeat(1, N, 1,
                                                        1)  # [B, N, N, H]

        # weighted and masked softmax
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = adj.unsqueeze(-1) * torch.exp(alpha)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        alpha = F.dropout(alpha, p=self.dropout,
                          training=self.training)  # [B, N, N, H]

        out = torch.matmul(alpha.movedim(3, 1),
                           out.movedim(2, 1)).movedim(1, 2)  # [B,N,H,C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

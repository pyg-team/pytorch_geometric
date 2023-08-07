import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor


class DenseGTVConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GTVConv`."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 delta_coeff: float = 1., eps: float = 1e-3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.delta_coeff = delta_coeff
        self.eps = eps

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, adj: Tensor,
                mask: OptTensor = None) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """

        # Update node features
        x = x @ self.weight

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        # Pairwise absolute differences
        abs_diff = torch.sum(torch.abs(x[..., None, :] - x[:, None, ...]),
                             dim=-1)

        # Gammma matrix
        mod_adj = adj / torch.clamp(abs_diff, min=1e-3)

        # Compute Laplacian L=D-A
        deg = torch.sum(mod_adj, dim=-1)
        mod_adj = torch.diag_embed(deg) - mod_adj

        # Compute modified laplacian: L_adjusted = I - delta*L
        mod_adj = torch.eye(N).to(x.device) - self.delta_coeff * mod_adj

        out = torch.matmul(mod_adj, x)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

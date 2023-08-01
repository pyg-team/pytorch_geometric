import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor


class DenseGTVConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GTVConv`."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 delta_coeff: float = 1., eps: float = 1e-3):
        super().__init__()

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.delta_coeff = delta_coeff
        self.eps = eps

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(
            self.weight
        )  # TODO: Maybe change to glorot as standard initializer
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

        # Absolute differences between neighbouring nodes
        batch_idx, node_i, node_j = torch.nonzero(adj, as_tuple=True)
        abs_diff = torch.sum(
            torch.abs(x[batch_idx, node_i, :] - x[batch_idx, node_j, :]),
            dim=-1)  # shape [B, E]

        # Gamma matrix
        mod_adj = torch.clone(adj)
        mod_adj[batch_idx, node_i,
                node_j] /= torch.clamp(abs_diff, min=self.eps)

        # Compute Laplacian L=D-A
        deg = torch.sum(mod_adj, dim=-1)
        mod_adj = -mod_adj
        mod_adj[:, range(N), range(N)] += deg

        # Compute modified laplacian: L_adjusted = I - delta*L
        mod_adj = -self.delta_coeff * mod_adj
        mod_adj[:, range(N), range(N)] += 1

        out = torch.matmul(mod_adj, x)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from torch_geometric.typing import OptTensor


class DenseSAGEConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.SAGEConv`.

    .. note::

        :class:`~torch_geometric.nn.dense.DenseSAGEConv` expects to work on
        binary adjacency matrices.
        If you want to make use of weighted dense adjacency matrices, please
        use :class:`torch_geometric.nn.dense.DenseGraphConv` instead.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, out_channels, bias=False)
        self.lin_root = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor,
                mask: OptTensor = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = self.lin_rel(out) + self.lin_root(x)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

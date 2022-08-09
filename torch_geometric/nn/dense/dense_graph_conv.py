import torch
from torch.nn import Linear


class DenseGraphConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(self, in_channels, out_channels, aggr='add', bias=True):
        assert aggr in ['add', 'mean', 'max']
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, adj, mask=None):
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
        B, N, C = x.size()

        if self.aggr in ['add', 'mean']:
            out = torch.matmul(adj, x)
            if self.aggr == 'mean':
                out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        elif self.aggr == 'max':
            out = x.unsqueeze(-2).repeat(1, 1, N, 1)
            adj = adj.unsqueeze(-1).expand(B, N, N, C)
            out[adj == 0] = float('-inf')
            out = out.max(dim=-3)[0]
            out[out == float('-inf')] = 0.

        out = self.lin_rel(out)
        out = out + self.lin_root(x)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

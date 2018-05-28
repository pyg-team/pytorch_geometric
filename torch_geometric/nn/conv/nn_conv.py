import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import degree

from ..inits import uniform


class NNConv(torch.nn.Module):
    """Convolution operator using a continuous kernel function computed by a
       neural network. The neural network kernel function maps
       pseudo-coordinates (adj) to in_channels x out_channels weights for the
       feature aggregation. (adapted from Gilmer et al.: Neural Message Passing
       for Quantum Chemistry, https://arxiv.org/abs/1704.01212)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (nn.Sequential): Neural network.
        norm (bool, optional): Whether to normalize output by node degree.
            (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`True`, the layer will
            add the weighted root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super(NNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.norm = norm

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

        for item in self.nn:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, pseudo):
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        out = self.nn(pseudo)
        out = out.view(-1, self.in_channels, self.out_channels)

        out = torch.matmul(x[col].unsqueeze(1), out).squeeze(1)
        out = scatter_add(out, row, dim=0)

        # Normalize by node degree (if wished).
        if self.norm:
            deg = degree(row, x.size(0), x.dtype, x.device)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

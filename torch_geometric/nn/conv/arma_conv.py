import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing

from ..inits import glorot, zeros


class ARMAConv(MessagePassing):
    r"""The ARMA graph convolutional operator from the `"Graph Neural Networks
    with Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_ paper

    .. math::
        \mathbf{X}_k^{(t+1)} = \mathbf{\hat{L}} \mathbf{X}_k^{(t)}
        \mathbf{W} + \mathbf{X}^{(0)} \mathbf{V},

    where :math:`\mathbf{\hat{L}} = \mathbf{I} - \mathbf{L} = \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}` denotes the
    modified Laplacian :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}`.

    Args:
        in_channels (int): Size of each input sample :math:`\mathbf{x}^{(t)}`.
        out_channels (int): Size of each output sample
            :math:`\mathbf{x}^{(t+1)}`.
        initial_channels (int): Size of each initial sample
            :math:`\mathbf{x}^{(0)}`.
        num_stacks (int, optional): Number of parallel stacks :math:`K`.
            (default: :obj:`1`).
        dropout (float, optional): Dropout probability of the skip connection.
            (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_channels,
                 num_stacks=1,
                 dropout=0,
                 bias=True):
        super(ARMAConv, self).__init__()

        self.initial_channels = initial_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.dropout = dropout

        self.w = Parameter(Tensor(num_stacks, in_channels, out_channels))
        self.v = Parameter(Tensor(num_stacks, initial_channels, out_channels))

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.w)
        glorot(self.v)
        zeros(self.bias)

    def forward(self, x, x_init, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x_init = x_init.unsqueeze(1) if x_init.dim() == 2 else x_init

        if edge_weight is None:
            edge_weight = x.new_ones((edge_index.size(1), ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        lap = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x.transpose(0, 1), self.w).transpose(1, 0)
        x_init = torch.matmul(x_init.transpose(0, 1), self.v).transpose(1, 0)
        return self.propagate('add', edge_index, x=x, x_init=x_init, lap=lap)

    def message(self, x_j, lap):
        return lap.view(-1, 1, 1) * x_j

    def update(self, aggr_out, x_init):
        if self.training and self.dropout > 0:
            x_init = F.dropout(x_init, p=self.dropout, training=True)

        out = aggr_out + x_init

        if self.num_stacks == 1:
            out = out.squeeze(1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, num_stacks={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_stacks)

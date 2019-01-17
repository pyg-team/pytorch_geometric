import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from ..inits import glorot, zeros


class ARMAConv(torch.nn.Module):
    r"""The ARMA graph convolutional operator from the `"Graph Neural Networks
    with Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^K \mathbf{X}_k^{(T)},

    with :math:`\mathbf{X}_k^{(T)}` being recursively defined by

    .. math::
        \mathbf{X}_k^{(t+1)} = \sigma \left( \mathbf{\hat{L}}
        \mathbf{X}_k^{(t)} \mathbf{W} + \mathbf{X}^{(0)} \mathbf{V} \right),

    where :math:`\mathbf{\hat{L}} = \mathbf{I} - \mathbf{L} = \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}` denotes the
    modified Laplacian :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}`.

    Args:
        in_channels (int): Size of each input sample :math:`\mathbf{x}^{(t)}`.
        out_channels (int): Size of each output sample
            :math:`\mathbf{x}^{(t+1)}`.
        num_stacks (int, optional): Number of parallel stacks :math:`K`.
            (default: :obj:`1`).
        num_layers (int, optional): Number of layers :math:`T`.
            (default: :obj:`1`)
        act (callable, optional): Activiation function :math:`\sigma`.
            (default: :meth:`torch.nn.functional.ReLU`)
        shared_weights (int, optional): If set to :obj:`True` the layers in
            each stack will share the same parameters. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the skip connection.
            (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_stacks=1,
                 num_layers=1,
                 shared_weights=False,
                 act=F.relu,
                 dropout=0,
                 bias=True):
        super(ARMAConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.act = act
        self.shared_weights = shared_weights
        self.dropout = dropout

        w = Parameter(torch.Tensor(num_stacks, in_channels, out_channels))
        self.ws = torch.nn.ParameterList([w])
        self.vs = torch.nn.ParameterList([])
        if shared_weights:
            w = Parameter(torch.Tensor(num_stacks, out_channels, out_channels))
            for i in range(num_layers - 1):
                self.ws.append(w)
            v = Parameter(torch.Tensor(num_stacks, in_channels, out_channels))
            for i in range(num_layers):
                self.vs.append(v)
        else:
            for i in range(self.num_layers - 1):
                w = Parameter(Tensor(num_stacks, out_channels, out_channels))
                self.ws.append(w)
            for i in range(self.num_layers):
                v = Parameter(Tensor(num_stacks, in_channels, out_channels))
                self.vs.append(v)

        if bias:
            self.bias = torch.nn.ParameterList([])
            if shared_weights:
                bias = Parameter(torch.Tensor(num_stacks, 1, out_channels))
                for i in range(num_layers):
                    self.bias.append(bias)
            else:
                for i in range(num_layers):
                    bias = Parameter(torch.Tensor(num_stacks, 1, out_channels))
                    self.bias.append(bias)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.ws:
            glorot(w)
        for v in self.vs:
            glorot(v)
        if self.bias is not None:
            for bias in self.bias:
                zeros(bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if edge_weight is None:
            edge_weight = x.new_ones((edge_index.size(1), ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        lap = deg_inv[row] * edge_weight * deg_inv[col]

        x, x_init = x.unsqueeze(0), x.unsqueeze(0)
        for i, (w, v) in enumerate(zip(self.ws, self.vs)):
            x = torch.matmul(x, w)
            x = x[:, col] * lap.view(1, -1, 1)
            x = scatter_add(x, row, dim=1, dim_size=x_init.size(1))

            skip = torch.matmul(x_init, v)
            if self.training and self.dropout > 0:
                skip = F.dropout(skip, p=self.dropout, training=True)

            x = x + skip

            if self.bias is not None:
                x = x + self.bias[i]

            if self.act is not None and i < self.num_layers - 1:
                x = self.act(x)

        return x.sum(dim=0)

    def __repr__(self):
        return '{}({}, {}, num_stacks={}, num_layers={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_stacks, self.num_layers)

import torch
from torch.nn import Parameter
import torch.nn.functional as F
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

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels

        w = torch.nn.Parameter(torch.Tensor(K, F_in, F_out))
        self.ws = torch.nn.ParameterList([w])
        for i in range(min(1, T - 1) if shared_weights else T - 1):
            self.ws.append(Parameter(torch.Tensor(K, F_out, F_out)))

        self.vs = torch.nn.ParameterList([])
        for i in range(1 if shared_weights else T):
            self.vs.append(Parameter(torch.Tensor(K, F_in, F_out)))

        if bias:
            self.bias = torch.nn.ParameterList([])
            for i in range(1 if shared_weights else T):
                self.bias.append(Parameter(torch.Tensor(K, 1, F_out)))
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

        lap = -deg_inv[row] * edge_weight * deg_inv[col]

        x = x.unsqueeze(0)
        out = x
        for t in range(self.num_layers):
            w = self.ws[min(t, 1) if self.shared_weights else t]
            out = torch.matmul(out, w)
            if self.training and self.dropout > 0:
                out = F.dropout(out, p=self.dropout, training=True)
            out = out[:, col] * lap.view(1, -1, 1)
            out = scatter_add(out, row, dim=1, dim_size=x.size(1))

            skip = x
            if self.training and self.dropout > 0:
                skip = F.dropout(skip, p=self.dropout, training=True)
            v = self.vs[0 if self.shared_weights else t]
            skip = torch.matmul(skip, v)

            out = out + skip

            if self.bias is not None:
                bias = self.bias[0 if self.shared_weights else t]
                out = out + bias

            if self.act:
                out = self.act(out)

        return out.mean(dim=0)

    def __repr__(self):
        return '{}({}, {}, num_stacks={}, num_layers={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_stacks, self.num_layers)

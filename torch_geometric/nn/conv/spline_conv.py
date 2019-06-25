import warnings

import torch
import torch_geometric
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.repeat import repeat

from ..inits import uniform, zeros

try:
    from torch_spline_conv import SplineBasis, SplineWeighting
except ImportError:
    SplineBasis = None
    SplineWeighting = None


class SplineConv(MessagePassing):
    r"""The spline-based convolutional operator from the `"SplineCNN: Fast
    Geometric Deep Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in
        \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a kernel function defined
    over the weighted B-Spline tensor product basis.

    .. note::

        Pseudo-coordinates must lay in the fixed interval :math:`[0, 1]` for
        this method to work as intended.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): If set to :obj:`False`, the
            operator will use a closed B-spline basis in this dimension.
            (default :obj:`True`)
        degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 aggr='mean',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(SplineConv, self).__init__(aggr=aggr, **kwargs)

        if SplineBasis is None:
            raise ImportError('`SplineConv` requires `torch-spline-conv`.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.degree = degree

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        K = kernel_size.prod().item()
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

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
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, pseudo):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        if not x.is_cuda:
            warnings.warn('We do not recommend using the non-optimized CPU '
                          'version of SplineConv. If possible, please convert '
                          'your data to the GPU first.')

        if torch_geometric.is_debug_enabled():
            if x.size(1) != self.in_channels:
                raise RuntimeError(
                    'Expected {} node features, but found {}'.format(
                        self.in_channels, x.size(1)))

            if pseudo.size(1) != self.dim:
                raise RuntimeError(
                    ('Expected pseudo-coordinate dimensionality of {}, but '
                     'found {}').format(self.dim, pseudo.size(1)))

            min_index, max_index = edge_index.min(), edge_index.max()
            if min_index < 0 or max_index > x.size(0) - 1:
                raise RuntimeError(
                    ('Edge indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         x.size(0) - 1, min_index, max_index))

            min_pseudo, max_pseudo = pseudo.min(), pseudo.max()
            if min_pseudo < 0 or max_pseudo > 1:
                raise RuntimeError(
                    ('Pseudo-coordinates must lay in the fixed interval [0, 1]'
                     ' but found them in the interval [{}, {}]').format(
                         min_pseudo, max_pseudo))

        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        data = SplineBasis.apply(pseudo, self._buffers['kernel_size'],
                                 self._buffers['is_open_spline'], self.degree)
        return SplineWeighting.apply(x_j, self.weight, *data)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

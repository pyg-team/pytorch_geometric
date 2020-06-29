import warnings
from typing import Union, Tuple, List
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.repeat import repeat

from ..inits import uniform, zeros

try:
    from torch_spline_conv import spline_basis, spline_weighting
except ImportError:
    spline_basis = None
    spline_weighting = None


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
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
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
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 dim: int,
                 kernel_size: Union[int, List[int]],
                 is_open_spline: bool = True,
                 degree: int = 1,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True,
                 **kwargs):  # yapf: disable
        super(SplineConv, self).__init__(aggr=aggr, **kwargs)

        if spline_basis is None:
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

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        K = kernel_size.prod().item()
        self.weight = Parameter(torch.Tensor(K, in_channels[0], out_channels))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0) * self.weight.size(1)
        uniform(size, self.weight)
        uniform(size, self.root)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if not x[0].is_cuda:
            warnings.warn(
                'We do not recommend using the non-optimized CPU version of '
                '`SplineConv`. If possible, please move your data to GPU.')

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        data = spline_basis(edge_attr, self.kernel_size, self.is_open_spline,
                            self.degree)
        return spline_weighting(x_j, self.weight, *data)

    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.in_channels, self.out_channels,
                                           self.dim)

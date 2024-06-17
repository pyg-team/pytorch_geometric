import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Parameter

import torch_geometric.typing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import uniform, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils.repeat import repeat

if torch_geometric.typing.WITH_TORCH_SPLINE_CONV:
    from torch_spline_conv import spline_basis, spline_weighting
else:
    spline_basis = spline_weighting = None


class SplineConv(MessagePassing):
    r"""The spline-based convolutional operator from the `"SplineCNN: Fast
    Geometric Deep Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper.

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
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): If set to :obj:`False`, the
            operator will use a closed B-spline basis in this dimension.
            (default :obj:`True`)
        degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
        aggr (str, optional): The aggregation scheme to use
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
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        dim: int,
        kernel_size: Union[int, List[int]],
        is_open_spline: bool = True,
        degree: int = 1,
        aggr: str = 'mean',
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        if spline_basis is None:
            raise ImportError("'SplineConv' requires 'torch-spline-conv'")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.degree = degree
        self.root_weight = root_weight

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.K = kernel_size.prod().item()

        if in_channels[0] > 0:
            self.weight = Parameter(
                torch.empty(self.K, in_channels[0], out_channels))
        else:
            self.weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if root_weight:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer='uniform')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if not isinstance(self.weight, nn.UninitializedParameter):
            size = self.weight.size(0) * self.weight.size(1)
            uniform(size, self.weight)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if not x[0].is_cuda:
            warnings.warn(
                'We do not recommend using the non-optimized CPU version of '
                '`SplineConv`. If possible, please move your data to GPU.')

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root_weight:
            out = out + self.lin(x_r)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        data = spline_basis(edge_attr, self.kernel_size, self.is_open_spline,
                            self.degree)
        return spline_weighting(x_j, self.weight, *data)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, torch.nn.parameter.UninitializedParameter):
            x = input[0][0] if isinstance(input, tuple) else input[0]
            in_channels = x.size(-1)
            self.weight.materialize((self.K, in_channels, self.out_channels))
            size = self.weight.size(0) * self.weight.size(1)
            uniform(size, self.weight)
        module._hook.remove()
        delattr(module, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.dim})')

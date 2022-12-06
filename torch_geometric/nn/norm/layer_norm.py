import torch
import torch.nn.functional as F
# from pyg_lib.ops import broadcast_sub
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter

from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree

from ..inits import ones, zeros


class LayerNorm(torch.nn.Module):
    r"""Applies layer normalization over each individual example in a batch
    of node features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optinal): The normalization mode to use for layer
            normalization. (:obj:`"graph"` or :obj:`"node"`). If :obj:`"graph"`
            is used, each graph will be considered as an element to be
            normalized. If `"node"` is used, each node will be considered as
            an element to be normalized. (default: :obj:`"graph"`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5,
                 affine: bool = True, mode: str = 'graph'):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = Parameter(torch.Tensor(in_channels))
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if self.mode == 'graph':
            if batch is None:
                x = x - x.mean()
                out = x / (x.std(unbiased=False) + self.eps)

            else:
                batch_size = int(batch.max()) + 1

                norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
                norm = norm.mul_(x.size(-1)).view(-1, 1)

                mean = scatter(x, batch, dim=0, dim_size=batch_size,
                               reduce='add').sum(dim=-1, keepdim=True) / norm

                x = x - mean.index_select(0, batch)

                var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                              reduce='add').sum(dim=-1, keepdim=True)
                var = var / norm

                out = x / (var + self.eps).sqrt().index_select(0, batch)

            if self.weight is not None and self.bias is not None:
                out = out * self.weight + self.bias

            return out

        if self.mode == 'node':
            return F.layer_norm(x, (self.in_channels, ), self.weight,
                                self.bias, self.eps)

        raise ValueError(f"Unknow normalization mode: {self.mode}")

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'affine={self.affine}, mode={self.mode})')


class HeteroLayerNorm(torch.nn.Module):
    def __init__(self, in_channels: int, num_types: int, eps: float = 1e-5,
                 affine: bool = True, mode: str = 'graph'):
        super().__init__()

        self.num_types = num_types
        self.fused_aggr = FusedAggregation(['mean', 'std'])

        if affine:
            self.weight = Parameter(torch.Tensor(num_types, in_channels))
            self.bias = Parameter(torch.Tensor(num_types, in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.fused_aggr.reset_parameters()
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, type_vec: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""
        # x = x.view(10, -1, 64)
        # with torch.no_grad():
        #     mean = x.mean(dim=1, keepdim=True)
        #     std = x.std(dim=1, keepdim=True)
        # out = (x - mean) / std
        # return out.view(-1, 64)

        with torch.no_grad():
            mean, std = self.fused_aggr(x, type_vec, dim_size=self.num_types)
            # mean = mean[type_vec]
            # std = std[type_vec]

        # return (x - mean) / std

        out = torch.ops.pyg.sampled_op(x, mean, None, type_vec, "sub")
        out = torch.ops.pyg.sampled_op(out, std, None, type_vec, "div")
        return out

        # return broadcast_sub(broadcast_sub(x, mean, type_vec), std, type_vec)

        # return (x - mean)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'num_types={self.num_types}, affine={self.affine}, '
                f'mode={self.mode})')

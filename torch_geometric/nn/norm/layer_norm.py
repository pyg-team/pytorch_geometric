from torch_geometric.typing import OptTensor

import torch
from torch.nn import Parameter
from torch import Tensor
from torch_scatter import scatter
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
    """
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor([in_channels]))
            self.bias = Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class MeanAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='mean')


class SumAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')


class MaxAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='max')


class MinAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='min')


class MulAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        # TODO Currently, `mul` reduction can only operate on `index`:
        self.assert_index_present(index)
        return self.reduce(x, index, None, dim_size, dim, reduce='mul')


class VarAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_2 = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        return mean_2 - mean * mean


class StdAggregation(Aggregation):
    def __init__(self):
        super().__init__()
        self.var_aggr = VarAggregation()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        var = self.var_aggr(x, index, ptr, dim_size, dim)
        return torch.sqrt(var.relu() + 1e-5)


class SoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper

    .. math::
        \mathrm{softmax}(\{ \mathbf{x}_i : i \in |\mathcal{X}| \}, t)
        = \sum_{i \in |\mathcal{X}|} \frac{ \exp( t \cdot \mathbf{x}_i)}
        {\sum_{j \in |\mathcal{X}|} \exp(t \cdot \mathbf{x}_j)}
        \cdot \mathbf{x}_{i},

    where :math:`t` controls the softness of the softmax when aggregating over
    a set of features :math:`\mathcal{X}`.

    Args:
        t (float, optional): Initial inverse temperature for softmax
            aggregation. (default: :obj:`1.0`)
        learn (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`t` for softmax aggregation dynamically.
            (default: :obj:`False`)
        semi_grad (bool, optional): If set to :obj:`True`, will turn off
            gradient calculation during softmax computation. Therefore, only
            semi-gradient is used during backpropagation. Useful for saving
            memory when :obj:`t` is not learnable. (default: :obj:`False`)
    """
    def __init__(self, t: float = 1.0, learn: bool = False,
                 semi_grad: bool = False):
        # TODO Learn distinct `t` per channel.
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        self._init_t = t
        self.t = Parameter(torch.Tensor(1)) if learn else t
        self.learn = learn
        self.semi_grad = semi_grad
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_t)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        alpha = x
        if not isinstance(self.t, (int, float)) or self.t != 1:
            alpha = x * self.t
        if not self.learn and self.semi_grad:
            with torch.no_grad():
                alpha = softmax(alpha, index, ptr, dim_size, dim)
        else:
            alpha = softmax(alpha, index, ptr, dim_size, dim)
        return self.reduce(x * alpha, index, ptr, dim_size, dim, reduce='sum')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')


class PowerMeanAggregation(Aggregation):
    def __init__(self, p: float = 1.0, learn: bool = False):
        # TODO Learn distinct `p` per channel.
        super().__init__()
        self._init_p = p
        self.p = Parameter(torch.Tensor(1)) if learn else p
        self.learn = learn
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.p, Tensor):
            self.p.data.fill_(self._init_p)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        out = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        if isinstance(self.p, (int, float)) and self.p == 1:
            return out
        return out.clamp_(min=0, max=100).pow(1. / self.p)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')

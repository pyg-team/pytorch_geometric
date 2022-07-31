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


class QuantileAggregation(Aggregation):
    r"""Quantile aggregation operator.

    Args:
        q (float): A scalar in the range [0, 1].
        interpolation (str): Interpolation method applied if the quantile point
            :math:`q(n - 1)` lies between two values :math:`x_i \le x_{i+1}`,
            with :math:`i < q(n - 1) < i+1`. Can be one of the following:

            - :obj:`"lower"`: Returns the one with lowest value.

            - :obj:`"higher"`: Returns the one with highest value.

            - :obj:`"midpoint"`: Returns the average of the two values.

            - :obj:`"nearest"`: Returns the one nearest to the quantile point.

            - :obj:`"linear"` (default): Returns a linear combination of the
                two elments, as
                :math:`x_i + (x_{i+1} - x_i)\cdot(q(n - 1) - i)`.

        fill_value (float): Default value in the case no entry is
            found for a given index (default: :obj:`NaN`).
    """

    interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

    def __init__(self, q: float, interpolation: str = 'linear',
                 fill_value: float = float('nan')):
        if q > 1. or q < 0:
            raise ValueError("'q' must be in the range [0, 1].")
        if interpolation not in self.interpolations:
            raise ValueError(f"Invalid '{interpolation}' interpolation method."
                             f" Available interpolations:"
                             f"{self.interpolations}")

        super().__init__()
        self.q = q
        self.interpolation = interpolation
        self.fill_value = fill_value

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        self.assert_index_present(index)

        # compute the quantile points
        count = torch.bincount(index, minlength=dim_size)
        cumsum = torch.cumsum(count, dim=0)
        q_point = self.q * (count - 1) + cumsum - count

        # shape used for expansions
        # (1, ..., 1, -1, 1, ..., 1)
        shape = [1] * x.dim()
        shape[dim] = -1

        # two sorts: the first one on the value,
        # the second (stable) on the indices
        x, x_perm = torch.sort(x, dim=dim)
        index = index.view(*shape).expand_as(x)
        index = index.take_along_dim(x_perm, dim=dim)
        index, index_perm = torch.sort(index, dim=dim, stable=True)
        x = x.take_along_dim(index_perm, dim=dim)

        # compute the quantile interpolations
        if self.interpolation == 'lower':
            idx = torch.floor(q_point).long()
            quantile = x.index_select(dim, idx)
        elif self.interpolation == 'higher':
            idx = torch.ceil(q_point).long()
            quantile = x.index_select(dim, idx)
        elif self.interpolation == 'nearest':
            idx = torch.round(q_point).long()
            quantile = x.index_select(dim, idx)
        else:
            l_idx = torch.floor(q_point).long()
            r_idx = torch.ceil(q_point).long()
            l_med = x.index_select(dim, l_idx)
            r_med = x.index_select(dim, r_idx)

            if self.interpolation == 'linear':
                q_frac = torch.frac(q_point).view(*shape)
                quantile = l_med + (r_med - l_med) * q_frac
            else:  # 'midpoint'
                quantile = 0.5 * l_med + 0.5 * r_med

        # if the number of elements is 0, return 'nan'/fill_value
        out_mask = (count > 0).view(*shape)
        return torch.where(
            out_mask, quantile,
            torch.tensor([self.fill_value], dtype=quantile.dtype,
                         device=quantile.device))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(q={self.q}, '
                f'interpolation="{self.interpolation}", '
                f'fill_value={self.fill_value})')


class MedianAggregation(QuantileAggregation):
    r"""Median aggregation operator, as used in `"Understanding Structural
    Vulnerability in Graph Convolutional Networks"
    <https://www.ijcai.org/proceedings/2021/310>`_.

    .. note::
        If the median lies between two values, the lowest one is returned.
        To compute the midpoint (or other kind of interpolation) of the two
        values, use :class:`QuantileAggregation` instead.

    Args:
        fill_value (float, optional): Default value in the case no entry is
            found for a given index (default: :obj:`NaN`).
    """
    def __init__(self, fill_value: float = float('nan')):
        super().__init__(q=0.5, interpolation='lower', fill_value=fill_value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fill_value={self.fill_value})'


class SoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper

    .. math::
        \mathrm{softmax}(\mathcal{X}|t) = \sum_{\mathbf{x}_i\in\mathcal{X}}
        \frac{\exp(t\cdot\mathbf{x}_i)}{\sum_{\mathbf{x}_j\in\mathcal{X}}
        \exp(t\cdot\mathbf{x}_j)}\cdot\mathbf{x}_{i},

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
    r"""The powermean aggregation operator based on a power term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper

    .. math::
        \mathrm{powermean}(\mathcal{X}|p) = \left(\frac{1}{|\mathcal{X}|}
        \sum_{\mathbf{x}_i\in\mathcal{X}}\mathbf{x}_i^{p}\right)^{1/p},

    where :math:`p` controls the power of the powermean when aggregating over
    a set of features :math:`\mathcal{X}`.

    Args:
        p (float, optional): Initial power for powermean aggregation.
            (default: :obj:`1.0`)
        learn (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`p` for powermean aggregation dynamically.
            (default: :obj:`False`)
    """
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

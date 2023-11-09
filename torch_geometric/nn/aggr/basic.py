import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class SumAggregation(Aggregation):
    r"""An aggregation operator that sums up features across a set of elements.

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')


class MeanAggregation(Aggregation):
    r"""An aggregation operator that averages features across a set of
    elements.

    .. math::
        \mathrm{mean}(\mathcal{X}) = \frac{1}{|\mathcal{X}|}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='mean')


class MaxAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise maximum across a
    set of elements.

    .. math::
        \mathrm{max}(\mathcal{X}) = \max_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='max')


class MinAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise minimum across a
    set of elements.

    .. math::
        \mathrm{min}(\mathcal{X}) = \min_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='min')


class MulAggregation(Aggregation):
    r"""An aggregation operator that multiples features across a set of
    elements.

    .. math::
        \mathrm{mul}(\mathcal{X}) = \prod_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        # TODO Currently, `mul` reduction can only operate on `index`:
        self.assert_index_present(index)
        return self.reduce(x, index, None, dim_size, dim, reduce='mul')


class VarAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise variance across a
    set of elements.

    .. math::
        \mathrm{var}(\mathcal{X}) = \mathrm{mean}(\{ \mathbf{x}_i^2 : x \in
        \mathcal{X} \}) - \mathrm{mean}(\mathcal{X})^2.

    Args:
        semi_grad (bool, optional): If set to :obj:`True`, will turn off
            gradient calculation during :math:`E[X^2]` computation. Therefore,
            only semi-gradients are used during backpropagation. Useful for
            saving memory and accelerating backward computation.
            (default: :obj:`False`)
    """
    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.semi_grad = semi_grad

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        if self.semi_grad:
            with torch.no_grad():
                mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
        else:
            mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
        return mean2 - mean * mean


class StdAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise standard deviation
    across a set of elements.

    .. math::
        \mathrm{std}(\mathcal{X}) = \sqrt{\mathrm{var}(\mathcal{X})}.

    Args:
        semi_grad (bool, optional): If set to :obj:`True`, will turn off
            gradient calculation during :math:`E[X^2]` computation. Therefore,
            only semi-gradients are used during backpropagation. Useful for
            saving memory and accelerating backward computation.
            (default: :obj:`False`)
    """
    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.var_aggr = VarAggregation(semi_grad)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        var = self.var_aggr(x, index, ptr, dim_size, dim)
        # Allow "undefined" gradient at `sqrt(0.0)`:
        out = var.clamp(min=1e-5).sqrt()
        out = out.masked_fill(out <= math.sqrt(1e-5), 0.0)
        return out


class SoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

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
            semi-gradients are used during backpropagation. Useful for saving
            memory and accelerating backward computation when :obj:`t` is not
            learnable. (default: :obj:`False`)
        channels (int, optional): Number of channels to learn from :math:`t`.
            If set to a value greater than :obj:`1`, :math:`t` will be learned
            per input feature channel. This requires compatible shapes for the
            input to the forward calculation. (default: :obj:`1`)
    """
    def __init__(self, t: float = 1.0, learn: bool = False,
                 semi_grad: bool = False, channels: int = 1):
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_t = t
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        self.t = Parameter(torch.empty(channels)) if learn else t
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_t)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        t = self.t
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            assert isinstance(t, Tensor)
            t = t.view(-1, self.channels)

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = x * t

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
    <https://arxiv.org/abs/2006.07739>`_ paper.

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
        channels (int, optional): Number of channels to learn from :math:`p`.
            If set to a value greater than :obj:`1`, :math:`p` will be learned
            per input feature channel. This requires compatible shapes for the
            input to the forward calculation. (default: :obj:`1`)
    """
    def __init__(self, p: float = 1.0, learn: bool = False, channels: int = 1):
        super().__init__()

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_p = p
        self.learn = learn
        self.channels = channels

        self.p = Parameter(torch.empty(channels)) if learn else p
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.p, Tensor):
            self.p.data.fill_(self._init_p)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        p = self.p
        if self.channels != 1:
            assert isinstance(p, Tensor)
            self.assert_two_dimensional_input(x, dim)
            p = p.view(-1, self.channels)

        if not isinstance(p, (int, float)) or p != 1:
            x = x.clamp(min=0, max=100).pow(p)

        out = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')

        if not isinstance(p, (int, float)) or p != 1:
            out = out.clamp(min=0, max=100).pow(1. / p)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')

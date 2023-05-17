from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr.fused import FusedAggregation


class BatchNorm(torch.nn.Module):
    r"""Applies batch normalization over a batch of features as described in
    the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all nodes
    inside the mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
        allow_single_element (bool, optional): If set to :obj:`True`, batches
            with only a single element will work as during in evaluation.
            That is the running mean and variance will be used.
            Requires :obj:`track_running_stats=True`. (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        allow_single_element: bool = False,
    ):
        super().__init__()

        if allow_single_element and not track_running_stats:
            raise ValueError("'allow_single_element' requires "
                             "'track_running_stats' to be set to `True`")

        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)
        self.in_channels = in_channels
        self.allow_single_element = allow_single_element

    def reset_running_stats(self):
        r"""Resets all running statistics of the module."""
        self.module.reset_running_stats()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
        """
        if self.allow_single_element and x.size(0) <= 1:
            return torch.nn.functional.batch_norm(
                x,
                self.module.running_mean,
                self.module.running_var,
                self.module.weight,
                self.module.bias,
                False,  # bn_training
                0.0,  # momentum
                self.module.eps,
            )
        return self.module(x)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'


class HeteroBatchNorm(torch.nn.Module):
    r"""Applies batch normalization over a batch of heterogeneous features as
    described in the `"Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper.
    Compared to :class:`BatchNorm`, :class:`HeteroBatchNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_types, in_channels))
            self.bias = Parameter(torch.Tensor(num_types, in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean',
                                 torch.Tensor(num_types, in_channels))
            self.register_buffer('running_var',
                                 torch.Tensor(num_types, in_channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.mean_var = FusedAggregation(['mean', 'var'])

        self.reset_parameters()

    def reset_running_stats(self):
        r"""Resets all running statistics of the module."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor): A vector that maps each entry to a type.
        """
        if not self.training and self.track_running_stats:
            mean, var = self.running_mean, self.running_var
        else:
            with torch.no_grad():
                mean, var = self.mean_var(x, type_vec, dim_size=self.num_types)

        if self.training and self.track_running_stats:
            if self.momentum is None:
                self.num_batches_tracked.add_(1)
                exp_avg_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exp_avg_factor = self.momentum

            with torch.no_grad():  # Update running mean and variance:
                type_index = torch.unique(type_vec)

                self.running_mean[type_index] = (
                    (1.0 - exp_avg_factor) * self.running_mean[type_index] +
                    exp_avg_factor * mean[type_index])
                self.running_var[type_index] = (
                    (1.0 - exp_avg_factor) * self.running_var[type_index] +
                    exp_avg_factor * var[type_index])

        out = (x - mean[type_vec]) / var.clamp(self.eps).sqrt()[type_vec]

        if self.affine:
            out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'num_types={self.num_types})')

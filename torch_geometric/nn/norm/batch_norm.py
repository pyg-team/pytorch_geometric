from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter, init
from torch_scatter import scatter_add


class BatchNorm(torch.nn.Module):
    r"""Applies batch normalization over a batch of node features as described
    in the `"Batch Normalization: Accelerating Deep Network Training by
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
    def __init__(self, in_channels: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True,
                 allow_single_element: bool = False):
        super().__init__()

        if allow_single_element and not track_running_stats:
            raise ValueError("'allow_single_element' requires "
                             "'track_running_stats' to be set to `True`")

        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)
        self.in_channels = in_channels
        self.allow_single_element = allow_single_element

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
    r"""Applies batch normalization over a batch of node features as described
    in the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma_{\tau} + \beta_{\tau}

    The mean and standard-deviation are calculated per-dimension individually
    for each node type inside the mini-batch.
    \gamma_{\tau} and \beta_{\tau} are learnable affine transform parameters,
    where {\tau} is the type of the node. There is a distint \gamma and \beta for
    each type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (torch.Tensor): Number of node types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        elementwise_affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """
    def __init__(self, in_channels: int, num_types: int,
                 elementwise_affine: bool = True,
                 track_running_stats: bool = True, eps: float = 1e-5,
                 momentum: float = 0.1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HeteroBatchNorm, self).__init__()
        self.device = device
        self.elementwise_affine = elementwise_affine
        self.track_running_stats = track_running_stats
        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.momentum = momentum

        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty((self.num_types, self.in_channels),
                            **factory_kwargs))
            self.bias = Parameter(
                torch.empty((self.num_types, self.in_channels),
                            **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.training and self.track_running_stats:
            self.register_buffer(
                'running_mean',
                torch.zeros((self.num_types, self.in_channels),
                            **factory_kwargs))
            self.register_buffer(
                'running_var',
                torch.ones((self.num_types, self.in_channels),
                           **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                'num_batches_tracked',
                torch.tensor(
                    0, dtype=torch.long, **{
                        k: v
                        for k, v in factory_kwargs.items() if k != 'dtype'
                    }))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def compute_mean_var(self, x, types):
        tmp = scatter_add(src=x, index=types, dim=0)
        count = scatter_add(torch.ones((x.shape[0], ), device=self.device),
                            types, dim=0).view(-1, 1)
        mean = tmp.div_(count.clamp(min=1))
        var = x - mean[types]
        var = var.mul_(var)
        var = scatter_add(src=var, index=types, dim=0)
        var = var.div_(count.clamp(min=1))

        return mean, var

    def forward(self, x, types):
        if self.track_running_stats:
            # if track running stats calculate mean and var in nograd env
            with torch.no_grad():
                mean, var = self.compute_mean_var(x, types)
            if self.training:
                if self.momentum is None:
                    self.num_batches_tracked.add_(1)
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

                # update running mean and var
                with torch.no_grad():
                    present_types = torch.unique(types)
                    self.running_mean[present_types] = (
                        1 - exponential_average_factor) * self.running_mean[
                            present_types] + exponential_average_factor * mean
                    self.running_var[present_types] = (
                        1 - exponential_average_factor) * self.running_var[
                            present_types] + exponential_average_factor * var
            mean = self.running_mean
            var = self.running_var

        else:
            mean, var = self.compute_mean_var(x, types)

        # compute out
        out = (x - mean[types]).div_(torch.sqrt(var + self.eps)[types])

        if self.elementwise_affine:
            out = out.mul_(self.weight[types]).add_(self.bias[types])

        return out

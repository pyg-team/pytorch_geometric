from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.instancenorm import _InstanceNorm

from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree, scatter


class InstanceNorm(_InstanceNorm):
    r"""Applies instance normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for
    each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
        device (torch.device, optional): The device to use for the module.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_channels, eps, momentum, affine,
                         track_running_stats, device=device)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        if batch is None:
            out = F.instance_norm(
                x.t().unsqueeze(0), self.running_mean, self.running_var,
                self.weight, self.bias, self.training
                or not self.track_running_stats, self.momentum, self.eps)
            return out.squeeze(0).t()

        if batch_size is None:
            batch_size = int(batch.max()) + 1

        mean = var = unbiased_var = x  # Dummies.

        if self.training or not self.track_running_stats:
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.view(-1, 1)
            unbiased_norm = (norm - 1).clamp_(min=1)

            mean = scatter(x, batch, dim=0, dim_size=batch_size,
                           reduce='sum') / norm

            x = x - mean.index_select(0, batch)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                          reduce='sum')
            unbiased_var = var / unbiased_norm
            var = var / norm

            momentum = self.momentum
            if self.running_mean is not None:
                self.running_mean = (
                    1 - momentum) * self.running_mean + momentum * mean.mean(0)
            if self.running_var is not None:
                self.running_var = (
                    1 - momentum
                ) * self.running_var + momentum * unbiased_var.mean(0)
        else:
            if self.running_mean is not None:
                mean = self.running_mean.view(1, -1).expand(batch_size, -1)
            if self.running_var is not None:
                var = self.running_var.view(1, -1).expand(batch_size, -1)

            x = x - mean.index_select(0, batch)

        out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_features})'

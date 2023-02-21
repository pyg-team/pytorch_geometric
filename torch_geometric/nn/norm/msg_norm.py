import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class MessageNorm(torch.nn.Module):
    r"""Applies message normalization over the aggregated messages as described
    in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper

    .. math::

        \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_{i} + s \cdot
        {\| \mathbf{x}_i \|}_2 \cdot
        \frac{\mathbf{m}_{i}}{{\|\mathbf{m}_i\|}_2} \right)

    Args:
        learn_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor :math:`s` of message normalization.
            (default: :obj:`False`)
    """
    def __init__(self, learn_scale: bool = False):
        super().__init__()

        self.scale = Parameter(torch.Tensor([1.0]), requires_grad=learn_scale)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.scale.data.fill_(1.0)

    def forward(self, x: Tensor, msg: Tensor, p: float = 2.0) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            msg (torch.Tensor): The message tensor :math:`\mathbf{M}`.
            p (float, optional): The norm :math:`p` to use for normalization.
                (default: :obj:`2.0`)
        """
        msg = F.normalize(msg, p=p, dim=-1)
        x_norm = x.norm(p=p, dim=-1, keepdim=True)
        return msg * x_norm * self.scale

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(learn_scale={self.scale.requires_grad})')

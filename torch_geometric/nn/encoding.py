import math
from typing import Optional

import torch
from torch import Tensor

__all__ = classes = [
    'PositionalEncoding',
    'TemporalEncoding',
]


class PositionalEncoding(torch.nn.Module):
    r"""The positional encoding scheme from the `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ paper.

    .. math::

        PE(x)_{2 \cdot i} &= \sin(x / 10000^{2 \cdot i / d})

        PE(x)_{2 \cdot i + 1} &= \cos(x / 10000^{2 \cdot i / d})

    where :math:`x` is the position and :math:`i` is the dimension.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
        base_freq (float, optional): The base frequency of sinusoidal
            functions. (default: :obj:`1e-4`)
        granularity (float, optional): The granularity of the positions. If
            set to smaller value, the encoder will capture more fine-grained
            changes in positions. (default: :obj:`1.0`)
        device (torch.device, optional): The device of the module.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        out_channels: int,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with "
                             f"odd 'out_channels' (got {out_channels}).")

        self.out_channels = out_channels
        self.base_freq = base_freq
        self.granularity = granularity

        # Compute frequency values, avoiding MPS logspace limitation
        if device is not None and device.type == 'mps':
            # MPS doesn't support logspace, compute on CPU first
            frequency = torch.logspace(0, 1, out_channels // 2, base_freq,
                                       device='cpu')
        else:
            frequency = torch.logspace(0, 1, out_channels // 2, base_freq,
                                       device=device)

        self.register_buffer('frequency', frequency, persistent=True)

        self.reset_parameters()

        # Move the entire module to the target device if specified
        if device is not None:
            self.to(device)

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""  # noqa: D419
        x = x / self.granularity if self.granularity != 1.0 else x
        # The frequency buffer will be on the same device as the module
        # thanks to PyTorch's automatic device handling for persistent buffers
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class TemporalEncoding(torch.nn.Module):
    r"""The time-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.

    It first maps each entry to a vector with exponentially decreasing values,
    and then uses the cosine function to project all values to range
    :math:`[-1, 1]`.

    .. math::
        y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

    where :math:`d` defines the output feature dimension, and
    :math:`1 \leq i \leq d`.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
        device (torch.device, optional): The device of the module.
            (default: :obj:`None`)
    """
    def __init__(self, out_channels: int,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.out_channels = out_channels

        sqrt = math.sqrt(out_channels)
        weight = 1.0 / sqrt**torch.linspace(0, sqrt, out_channels,
                                            device=device).view(1, -1)
        self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""  # noqa: D419
        return torch.cos(x.view(-1, 1) @ self.weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'

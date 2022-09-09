from typing import Optional

import torch
from torch import Tensor


class PositionalEncoding1D(torch.nn.Module):
    r"""The 1D positional encoding from `"Attention Is All You Need"
    <https://arxiv.org/pdf/1706.03762.pdf>`_ paper.

    Args:
        out_channels (int): Size of positional encoding.
        max_position (int, optional): Maximum value of positions for
            :obj:`"learn"` mode. (default: :obj:`None`)
        base_freq (float, optional): The base frequncy of sinusoidal functions
            for :obj:`"sin"` mode. (default: :obj:`1e-4`)
        granularity (float, optional): The granularity of the positions for
            :obj:`"sin"` mode. If set to smaller, the :obj:`"sin"` encoder
            will capturing more fine-grained changes in position.
            (default: :obj:`1.0`)
        mode (str, optional): The mode to use for positonal encoding
            (:obj:`"sin"`, :obj:`"learn"`). (default: :obj:`sin`)
    """
    def __init__(
        self,
        out_channels: int,
        max_position: Optional[int] = None,
        base_freq: Optional[float] = 1e-4,
        granularity: Optional[float] = 1.0,
        mode: Optional[str] = "sin",
    ):
        super().__init__()

        self.mode = mode
        self.max_position = max_position
        self.out_channels = out_channels

        if self.mode == "sin":
            if out_channels % 2 != 0:
                raise ValueError(
                    "Cannot use sinusoidal positional encoding with "
                    f"odd `out_channles`(got {out_channels}).")

            self.frequency = torch.logspace(0, 1, steps=out_channels // 2,
                                            base=base_freq)
            self.granularity = granularity

        if self.mode == "learn":
            if max_position is None:
                raise ValueError(
                    "Need to specify `max_position` for `learn` mode.")

            self.position_encoding = torch.nn.Embedding(
                max_position,
                out_channels,
            )

    def compute_sinus_encoding(self, position: Tensor) -> Tensor:
        position_encoding = torch.zeros(
            (position.size(0), self.out_channels),
            device=position.device,
        )
        x = (position / self.granularity).unsqueeze(1) * self.frequency
        position_encoding[:, 0::2] = torch.sin(x)
        position_encoding[:, 1::2] = torch.cos(x)
        return position_encoding

    def forward(self, position: Tensor) -> Tensor:
        if self.mode == "sin":
            return self.compute_sinus_encoding(position)
        if self.mode == "learn":
            return self.position_encoding(position)
        raise ValueError(f"Encoding mode '{self.mode}' is not supported.")

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'max_position={self.max_position}, mode={self.mode})')

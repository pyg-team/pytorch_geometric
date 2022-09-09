from typing import Optional

import torch
from torch import Tensor


class PositionalEncoding1D(torch.nn.Module):
    r"""The 1D positional encoding from `"Attention Is All You Need"
    <https://arxiv.org/pdf/1706.03762.pdf>`_ paper.
    """
    def __init__(self, out_channels: int, max_position: Optional[int] = None,
                 max_inv_freq: int = 10000., mode: str = "sin"):
        super().__init__()

        self.mode = mode
        self.max_position = max_position
        self.out_channels = out_channels

        if self.mode == "sin":
            if out_channels % 2 != 0:
                raise ValueError(
                    "Cannot use sinusoidal positional encoding with "
                    f"odd `out_channles`(got {out_channels}).")

            self.frequency = 1.0 / (max_inv_freq**(torch.arange(
                0, out_channels, 2, dtype=torch.float) / out_channels))

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
        x = position.unsqueeze(1) * self.frequency
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

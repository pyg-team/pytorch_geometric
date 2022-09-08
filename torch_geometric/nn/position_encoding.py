from typing import Union

import torch
import torch.nn.functional as F


class PositionEncoding1D(torch.nn.Module):
    r"""The 1D positioning encoding from `"Attention Is All You Need"
    <https://arxiv.org/pdf/1706.03762.pdf>`_ paper.
    """
    def __init__(self, num_pos: int, out_channels: int,
                 max_wave_len: Union[float, int] = 10000., mode: str = "sin"):
        super().__init__()

        self.mode = mode
        self.num_pos = num_pos
        self.out_channles = out_channels

        if self.mode == "sin":
            if out_channels % 2 != 0:
                raise ValueError(
                    "Cannot use sinusoidal positional encoding with "
                    f"odd dim. Got `out_channles`: {out_channels}).")

            position_encoding = torch.zeros(num_pos, out_channels)

            position = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(1)
            frequency = 1.0 / (max_wave_len**(torch.arange(
                0, out_channels, 2, dtype=torch.float) / out_channels))

            x = position * frequency
            position_encoding[:, 0::2] = torch.sin(x)
            position_encoding[:, 1::2] = torch.cos(x)

            self.register_buffer('position_encoding', position_encoding)

        if self.mode == "learn":
            self.position_encoding = torch.nn.Embedding(num_pos, out_channels)

    def forward(self, pos_ids):
        if self.mode == "sin":
            return F.embedding(pos_ids, self.position_encoding)
        if self.mode == "learn":
            return self.position_encoding(pos_ids)
        raise ValueError(f"Encoding mode '{self.mode}' is not supported.")

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.num_pos}, {self.out_channles}, '
            f'mode={self.mode})')

from typing import Optional

import torch
from torch import Tensor


class MLPGraphHead(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels,
        reduce: str = "mean",
    ):
        pass

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """"""
        return x

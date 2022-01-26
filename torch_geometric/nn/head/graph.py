from typing import Optional

import torch
from torch import Tensor


class GraphHead(torch.nn.Module):
    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class MLPGraphHead(GraphHead):
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

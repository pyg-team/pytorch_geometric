from typing import Optional

import torch
from torch import Tensor


class IdentityNodeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return x


class MLPNodeHead(torch.nn.Module):
    def __init__(
        self,
        in_channels,
    ):
        pass

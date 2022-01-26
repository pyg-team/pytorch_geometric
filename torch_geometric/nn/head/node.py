import torch
from torch import Tensor


class NodeHead(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class IdentityNodeHead(NodeHead):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return x


class MLPNodeHead(NodeHead):
    def __init__(
        self,
        in_channels,
    ):
        pass

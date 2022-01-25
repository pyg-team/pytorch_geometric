import torch
from torch import Tensor


class ConcatLinkHead(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        pass

    def forward(self, x_src: Tensor, x_dst: Tensor) -> Tensor:
        """"""
        return self.MLP(torch.cat([x_src, x_dst], dim=-1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class InnerProductLinkHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_src: Tensor, x_dst: Tensor) -> Tensor:
        """"""
        return (x_src * x_dst).sum(dim=-1, keepdim=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

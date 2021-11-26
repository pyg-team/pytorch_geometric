import torch
from torch import Tensor


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = x.view(*self.shape)
        return x

    def __repr__(self) -> str:
        shape = ', '.join([str(dim) for dim in self.shape])
        return f'{self.__class__.__name__}({shape})'

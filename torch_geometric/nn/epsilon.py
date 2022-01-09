import torch
from torch import Tensor


class EPSILON(object):
    r"""A small value adjustable to the precision"""

    def __init__(self):
        super().__init__()

    def __add__(self, other: Tensor) -> Tensor:
        if other.dtype == torch.float16:
            return other + 1e-7
        else:
            return other + 1e-15

    def __radd__(self, other: Tensor) -> Tensor:
        return self.__add__(other)

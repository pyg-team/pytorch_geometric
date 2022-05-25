from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr


class BaseAggr(torch.nn.Module, ABC):
    r"""An abstract base class for writing aggregations."""
    @abstractmethod
    def forward(self, x: Tensor, index: Tensor, dim_size: Optional[int] = None,
                ptr: Optional[Tensor] = None, dim: int = -2) -> Tensor:
        pass

    def reset_parameters(self):
        pass

    def reduce(self, x: Tensor, index: Tensor, dim_size: Optional[int] = None,
               ptr: Optional[Tensor] = None, dim: int = -2,
               reduce: str = 'add') -> Tensor:

        if ptr is not None:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment_csr(x, ptr, reduce=reduce)

        return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def expand_left(src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr


class Aggregation(torch.nn.Module, ABC):
    r"""An abstract base class for writing aggregations."""
    @abstractmethod
    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        pass

    def reset_parameters(self):
        pass

    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'add') -> Tensor:

        assert index is not None or ptr is not None

        if ptr is not None:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment_csr(x, ptr, reduce=reduce)

        if index is not None:
            return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

        raise ValueError(f"Error in '{self.__class__.__name__}': "
                         f"Both 'index' and 'ptr' are undefined")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

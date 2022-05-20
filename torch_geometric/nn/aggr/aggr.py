from abc import ABC
from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr


class Aggr(torch.nn.Module, ABC):
    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim: int = -2,
        dim_size: Optional[None] = None,
    ) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class MeanAggr(Aggr):
    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim: int = -2,
        dim_size: Optional[None] = None,
    ) -> Tensor:
        if ptr is not None:
            ptr = expand_left(ptr, dim=dim, dims=x.dim())
            return segment_csr(x, ptr, reduce='mean')
        else:
            return scatter(x, index, dim=dim, dim_size=dim_size, reduce='mean')


###############################################################################


def expand_left(src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src

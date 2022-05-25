from abc import ABC
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter, scatter_softmax, segment_csr


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


class MaxAggr(Aggr):
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
            return segment_csr(x, ptr, reduce='max')
        else:
            return scatter(x, index, dim=dim, dim_size=dim_size, reduce='max')


class MinAggr(Aggr):
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
            return segment_csr(x, ptr, reduce='min')
        else:
            return scatter(x, index, dim=dim, dim_size=dim_size, reduce='min')


class SumAggr(Aggr):
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
            return segment_csr(x, ptr, reduce='sum')
        else:
            return scatter(x, index, dim=dim, dim_size=dim_size, reduce='sum')


class SoftmaxAggr(Aggr):
    def __init__(self, t: float = 1.0, learn: bool = False):
        super().__init__()
        self.init_t = t
        self.learn = learn
        if learn:
            self.t = Parameter(torch.Tensor([t]), requires_grad=True)
        else:
            self.t = t

    def reset_parameters(self):
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.init_t)

    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim: int = -2,
        dim_size: Optional[None] = None,
    ) -> Tensor:
        if ptr is not None:
            raise NotImplementedError
        else:
            if self.learn:
                w = scatter_softmax(x * self.t, index, dim=dim)
            else:
                with torch.no_grad():
                    w = scatter_softmax(x * self.t, index, dim=dim)
            return scatter(x * w, index, dim=dim, dim_size=dim_size,
                           reduce='add')


class PowermeanAggr(Aggr):
    def __init__(self, p: float = 1.0, learn: bool = False):
        super().__init__()
        self.init_p = p
        if learn:
            self.p = Parameter(torch.Tensor([p]), requires_grad=True)
        else:
            self.p = p

    def reset_parameters(self):
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.init_p)

    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim: int = -2,
        dim_size: Optional[None] = None,
    ) -> Tensor:
        x = torch.pow(torch.abs(x), self.p)
        if ptr is not None:
            ptr = expand_left(ptr, dim=dim, dims=x.dim())
            return torch.pow(segment_csr(x, ptr, reduce='mean'), 1 / self.p)
        else:
            return torch.pow(
                scatter(x, index, dim=dim, dim_size=dim_size, reduce='mean'),
                1 / self.p)


class VarAggr(Aggr):
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
            mean = segment_csr(x, ptr, reduce='mean')
            mean_squares = segment_csr(x * x, ptr, reduce='mean')
        else:
            mean = scatter(x, index, dim=dim, dim_size=dim_size, reduce='mean')
            mean_squares = scatter(x * x, index, dim=dim, dim_size=dim_size,
                                   reduce='mean')
        return mean_squares - mean * mean


class StdAggr(Aggr):
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
            mean = segment_csr(x, ptr, reduce='mean')
            mean_squares = segment_csr(x * x, ptr, reduce='mean')
        else:
            mean = scatter(x, index, dim=dim, dim_size=dim_size, reduce='mean')
            mean_squares = scatter(x * x, index, dim=dim, dim_size=dim_size,
                                   reduce='mean')
        return torch.sqrt(torch.relu(mean_squares - mean * mean) + 1e-5)


###############################################################################


def expand_left(src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src

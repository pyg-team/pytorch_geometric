from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr


class Aggregation(torch.nn.Module, ABC):
    r"""An abstract base class for implementing custom aggregations."""
    requires_sorted_index = False

    @abstractmethod
    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            index (torch.LongTensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or `ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.LongTensor, optional): If given, computes the
                aggregation based on sorted inputs in CSR representation.
                One of :obj:`index` or `ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
        """
        pass

    def reset_parameters(self):
        pass

    def __call__(self, x: Tensor, index: Optional[Tensor] = None, *,
                 ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                 dim: int = -2) -> Tensor:

        if index is None and ptr is None:
            raise ValueError(f"Expected that either 'index' or 'ptr' is "
                             f"passed to '{self.__class__.__name__}'")

        if (self.requires_sorted_index and index is not None
                and not torch.all(index[:-1] <= index[1:])):
            raise ValueError(f"Can not perform aggregation inside "
                             f"'{self.__class__.__name__}' since the "
                             f"'index' tensor is not sorted")

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if (ptr is not None and dim_size is not None
                and dim_size != ptr.numel() - 1):
            raise ValueError(f"Encountered mismatch between 'dim_size' (got "
                             f"'{dim_size}') and 'ptr' (got '{ptr.size(0)}')")

        return super().__call__(x, index, ptr=ptr, dim_size=dim_size, dim=dim)

    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'add') -> Tensor:

        if ptr is not None:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment_csr(x, ptr, reduce=reduce)

        assert index is not None
        return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

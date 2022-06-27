from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr

from torch_geometric.utils import to_dense_batch


class Aggregation(torch.nn.Module, ABC):
    r"""An abstract base class for implementing custom aggregations."""
    @abstractmethod
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
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

    def __call__(self, x: Tensor, index: Optional[Tensor] = None,
                 ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                 dim: int = -2) -> Tensor:

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=torch.long)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f"'{ptr.numel() - 1}')")

        if index is not None:
            if dim_size is None:
                dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
            elif index.numel() > 0 and dim_size <= int(index.max()):
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f">= '{int(index.max()) + 1}')")

        return super().__call__(x, index, ptr, dim_size, dim)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    # Assertions ##############################################################

    def assert_index_present(self, index: Optional[Tensor]):
        # TODO Currently, not all aggregators support `ptr`. This assert helps
        # to ensure that we require `index` to be passed to the computation:
        if index is None:
            raise NotImplementedError(f"'{self.__class__.__name__}' requires "
                                      f"'index' to be specified")

    def assert_sorted_index(self, index: Optional[Tensor]):
        if index is not None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError(f"Can not perform aggregation inside "
                             f"'{self.__class__.__name__}' since the "
                             f"'index' tensor is not sorted")

    def assert_two_dimensional_input(self, x: Tensor, dim: int):
        if x.dim() != 2:
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"two-dimensional inputs (got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"'{self.__class__.__name__}' needs to perform "
                             f"aggregation in first dimension (got '{dim}')")

    # Helper methods ##########################################################

    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'add') -> Tensor:

        if ptr is not None:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment_csr(x, ptr, reduce=reduce)

        assert index is not None
        return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

    def to_dense_batch(self, x: Tensor, index: Optional[Tensor] = None,
                       ptr: Optional[Tensor] = None,
                       dim_size: Optional[int] = None,
                       dim: int = -2) -> Tuple[Tensor, Tensor]:

        # TODO Currently, `to_dense_batch` can only operate on `index`:
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(x, index, batch_size=dim_size)


###############################################################################


def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

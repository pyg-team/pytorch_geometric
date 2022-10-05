from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr

from torch_geometric.utils import scatter, to_dense_batch


class Aggregation(torch.nn.Module):
    r"""An abstract base class for implementing custom aggregations.

    Aggregation can be either performed via an :obj:`index` vector, which
    defines the mapping from input elements to their location in the output:

    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Notably, :obj:`index` does not have to be sorted:

    .. code-block::

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Assign each element to one of three sets:
       index = torch.tensor([0, 0, 1, 0, 2, 0, 2, 1, 0, 2])

       output = aggr(x, index)  #  Output shape: [4, 64]

    Alternatively, aggregation can be achieved via a "compressed" index vector
    called :obj:`ptr`. Here, elements within the same set need to be grouped
    together in the input, and :obj:`ptr` defines their boundaries:

    .. code-block::

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Define the boundary indices for three sets:
       ptr = torch.tensor([0, 4, 7, 10])

       output = aggr(x, ptr=ptr)  #  Output shape: [4, 64]

    Note that at least one of :obj:`index` or :obj:`ptr` must be defined.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or edge features
          :math:`(|\mathcal{E}|, F_{in})`,
          index vector :math:`(|\mathcal{V}|)` or :math:`(|\mathcal{E}|)`,
        - **output:** graph features :math:`(|\mathcal{G}|, F_{out})` or node
          features :math:`(|\mathcal{V}|, F_{out})`
    """
    _validate = __debug__

    # @abstractmethod
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            index (torch.LongTensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.LongTensor, optional): If given, computes the
                aggregation based on sorted inputs in CSR representation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
        """
        pass

    def reset_parameters(self):
        pass

    @staticmethod
    def set_validate_args(value: bool):
        r"""Sets whether validation is enabled or disabled.

        The default behavior mimics Python's :obj:`assert`` statement:
        validation is on by default, but is disabled if Python is run in
        optimized mode (via :obj:`python -O`).
        Validation may be expensive, so you may want to disable it once a model
        is working.

        Args:
            value (bool): Whether to enable validation.
        """
        Aggregation._validate = value

    def __call__(self, x: Tensor, index: Optional[Tensor] = None,
                 ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                 dim: int = -2, **kwargs) -> Tensor:

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
            elif self._validate:
                if index.numel() > 0 and dim_size <= int(index.max()):
                    raise ValueError(f"Encountered invalid 'dim_size' (got "
                                     f"'{dim_size}' but expected "
                                     f">= '{int(index.max()) + 1}')")

        return super().__call__(x, index, ptr, dim_size, dim, **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    # Assertions ##############################################################

    def assert_index_present(self, index: Optional[Tensor]):
        # TODO Currently, not all aggregators support `ptr`. This assert helps
        # to ensure that we require `index` to be passed to the computation:
        if index is None:
            raise NotImplementedError(
                "Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: Optional[Tensor]):
        if index is not None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError("Can not perform aggregation since the 'index' "
                             "tensor is not sorted")

    def assert_two_dimensional_input(self, x: Tensor, dim: int):
        if x.dim() != 2:
            raise ValueError(f"Aggregation requires two-dimensional inputs "
                             f"(got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"Aggregation needs to perform aggregation in "
                             f"first dimension (got '{dim}')")

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
                       dim_size: Optional[int] = None, dim: int = -2,
                       fill_value: float = 0.) -> Tuple[Tensor, Tensor]:

        # TODO Currently, `to_dense_batch` can only operate on `index`:
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(x, index, batch_size=dim_size,
                              fill_value=fill_value)


###############################################################################


def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

import torch
from torch import Tensor

from torch_geometric.typing import Adj, SparseTensor


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (SparseTensor, Tensor, str) -> Tensor
    pass


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (Tensor, Tensor, str) -> Tensor
    pass


def spmm(src: Adj, other: Tensor, reduce: str = "sum",
         convert_to_csr: bool = False) -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        src (Tensor or torch_sparse.SparseTensor]): The input sparse matrix,
            either a :class:`torch_sparse.SparseTensor` or a
            :class:`torch.sparse.Tensor`.
        other (Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    assert reduce in [
        'sum', 'add', 'mean', 'min', 'max'
    ], (reduce, " reduction is not supported for `torch.sparse.Tensor`.")

    reduce = 'sum' if reduce == 'add' else reduce

    if isinstance(src, SparseTensor):
        src = src.to_torch_sparse_csr_tensor(dtype=other.dtype)

    if not src.layout == torch.sparse_csr:
        src = src.to_sparse_csr()

    return torch.sparse.mm(src, other, reduce)

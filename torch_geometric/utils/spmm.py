import torch
from torch import Tensor

from torch_geometric.typing import Adj, SparseTensor, torch_sparse
from torch_geometric.utils import is_torch_sparse_tensor


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (Tensor, Tensor, str) -> Tensor
    pass


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (SparseTensor, Tensor, str) -> Tensor
    pass


def spmm(src: Adj, other: Tensor, reduce: str = "sum") -> Tensor:
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
    assert reduce in ['sum', 'add', 'mean', 'min', 'max']

    if isinstance(src, SparseTensor):
        return torch_sparse.matmul(src, other, reduce)

    if not is_torch_sparse_tensor(src):
        raise ValueError("`src` must be a `torch_sparse.SparseTensor` "
                         f"or a `torch.sparse.Tensor` (got {type(src)}).")

    if reduce in ['sum', 'add']:
        return torch.sparse.mm(src, other)

    # TODO: Support `mean` reduction for PyTorch SparseTensor
    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"`torch.sparse.Tensor`.")

from typing import Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from .torch_sparse_tensor import is_torch_sparse_tensor


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (Tensor, Tensor, str) -> Tensor
    pass


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (SparseTensor, Tensor, str) -> Tensor
    pass


def spmm(
    src: Union[SparseTensor, Tensor],
    other: Tensor,
    reduce: str = "sum",
) -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        src (Union[SparseTensor, Tensor]): The input sparse matrix,
            either of :obj:`torch_sparse.SparseTensor` or
            `torch.sparse.Tensor`.
        other (Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"sum"`, :obj:`"add"`, :obj:`"mean"`, :obj:`"max"`,
            :obj:`"min"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    assert reduce in {"sum", "add", "mean", "max", "min"}
    if isinstance(src, SparseTensor):
        return src.spmm(other, reduce)
    if not is_torch_sparse_tensor(src):
        raise ValueError("`src` must be a torch_sparse SparseTensor "
                         f"or a PyTorch SparseTensor, but got {type(src)}.")
    if reduce in {"sum", "add"}:
        return torch.sparse.mm(src, other)
    elif reduce == "mean":
        # TODO: Support `mean` reduction for PyTorch SparseTensor
        raise NotImplementedError
    else:
        raise NotImplementedError("`max` and `min` reduction are not supported"
                                  "for PyTorch SparseTensor input.")

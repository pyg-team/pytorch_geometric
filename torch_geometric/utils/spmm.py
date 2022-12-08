import torch
from torch import Tensor
from torch_geometric.typing import Adj, SparseTensor
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

    assert reduce in [
        'sum', 'add', 'mean', 'min', 'max'
    ], f"Uknown reduction type {reduce}. Supported: ['sum','add', 'mean','max','min']"
    reduce = 'sum' if reduce == 'add' else reduce

    # TODO: When torch.sparse.Tensor is available and use more strict        is_torch_sparse_tensor(src)
    # if not is_torch_sparse_tensor(src):
    #     raise ValueError("`src` must be a `torch_sparse.SparseTensor` "
    #                      f"or a `torch.sparse.Tensor` (got {type(src)}).")

    if isinstance(src, SparseTensor):
        if other.requires_grad or src.requires_grad():
            row = src.storage.row()
            csr2csc = src.storage.csr2csc()
            ccol_indices = src.storage.colptr()
        csr = src.to_torch_sparse_csr_tensor(dtype=other.dtype)

    else:
        csr = src
        row = None
        csr2csc = None
        ccol_indices = None

    if not csr.layout == torch.sparse_csr:
        raise ValueError(
            f"src must be a `torch.Tensor` with `torch.sparse_csr` layout {csr.layout}"
        )

    if other.requires_grad or csr.requires_grad:
        return torch.sparse.spmm_reduce(csr, other, reduce, row, ccol_indices,
                                        csr2csc)
    else:
        return torch.sparse.spmm_reduce(csr, other, reduce)


SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)

import torch
from torch import Tensor
from torch_geometric.typing import Adj, SparseTensor, torch_sparse
from torch_geometric.utils import is_sparse


@torch.jit._overload
def spmm(src, other, reduce, convert_to_csr):
    # type: (SparseTensor, Tensor, str, bool) -> Tensor
    pass

@torch.jit._overload
def spmm(src, other, reduce, convert_to_csr):
    # type: (Tensor, Tensor, str, bool) -> Tensor
    pass


def spmm(src: Adj, other: Tensor, reduce: str = "sum", convert_to_csr: bool = False) -> Tensor:
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
    assert reduce in ['sum', 'add', 'mean', 'min', 'max'], (reduce, 
                    " reduction is not supported for `torch.sparse.Tensor`.")
    # if not is_sparse(src):
    #     raise ValueError("`src` must be a `torch_sparse.SparseTensor` "
    #                      f"or a `torch.sparse.Tensor` (got {type(src)}).")
    
    if isinstance(src, SparseTensor):
        return torch_sparse.matmul(src, other, reduce)
    
    if convert_to_csr:
        src = src.to_sparse_csr()
    else:
        if not (src.layout == torch.sparse_csr):
             raise ValueError("`src` must be a `torch.Tensor` in "
                         f"`torch.sparse_csr` format (got {src.layout}). "
                         "To auto-convert use `convert_to_csr=True`")
     
    return torch.sparse.mm(src, other, reduce)

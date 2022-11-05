import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.utils import is_torch_sparse_tensor


def spmm(src: SparseTensor, other: Tensor, reduce: str = "sum") -> Tensor:
    assert reduce in {"sum", "mean", "max", "min"}
    if isinstance(src, SparseTensor):
        return src.spmm(other, reduce)
    if not is_torch_sparse_tensor(src):
        raise ValueError("`src` must be a torch_sparse SparseTensor "
                         f"or a PyTorch SparseTensor, but got {type(src)}.")
    if reduce == "sum":
        return torch.sparse.mm(src, other)
    elif reduce == "mean":
        # TODO: Support `mean` reduction for PyTorch SparseTensor
        raise NotImplementedError
    else:
        raise NotImplementedError("`max` and `min` reduction are not supported"
                                  "for PyTorch SparseTensor input.")

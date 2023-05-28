import warnings

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import Adj, SparseTensor, torch_sparse
from torch_geometric.utils import is_torch_sparse_tensor, scatter


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
        src (torch.Tensor or torch_sparse.SparseTensor): The input sparse
            matrix, either a :pyg:`PyG` :class:`torch_sparse.SparseTensor` or a
            :pytorch:`PyTorch` :class:`torch.sparse.Tensor`.
        other (torch.Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    reduce = 'sum' if reduce == 'add' else reduce

    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError(f"`reduce` argument '{reduce}' not supported")

    if isinstance(src, SparseTensor):
        if (torch_geometric.typing.WITH_PT2 and other.dim() == 2
                and not src.is_cuda() and not src.requires_grad()):
            # Use optimized PyTorch `torch.sparse.mm` path:
            csr = src.to_torch_sparse_csr_tensor()
            return torch.sparse.mm(csr, other, reduce)
        return torch_sparse.matmul(src, other, reduce)

    if not is_torch_sparse_tensor(src):
        raise ValueError("`src` must be a `torch_sparse.SparseTensor` "
                         f"or a `torch.sparse.Tensor` (got {type(src)}).")

    # `torch.sparse.mm` only supports reductions on CPU for PyTorch>=2.0.
    # This will currently throw on error for CUDA tensors.
    if torch_geometric.typing.WITH_PT2:

        if src.is_cuda and (reduce == 'min' or reduce == 'max'):
            raise NotImplementedError(f"`{reduce}` reduction is not yet "
                                      f"supported for 'torch.sparse.Tensor' "
                                      f"on device '{src.device}'")

        # Always convert COO to CSR for more efficient processing:
        if src.layout == torch.sparse_coo:
            warnings.warn(f"Converting sparse tensor to CSR format for more "
                          f"efficient processing. Consider converting your "
                          f"sparse tensor to CSR format beforehand to avoid "
                          f"repeated conversion (got '{src.layout}')")
            src = src.to_sparse_csr()

        # Warn in case of CSC format without gradient computation:
        if src.layout == torch.sparse_csc and not other.requires_grad:
            warnings.warn(f"Converting sparse tensor to CSR format for more "
                          f"efficient processing. Consider converting your "
                          f"sparse tensor to CSR format beforehand to avoid "
                          f"repeated conversion (got '{src.layout}')")

        # Use the default code path for `sum` reduction (works on CPU/GPU):
        if reduce == 'sum':
            return torch.sparse.mm(src, other)

        # Use the default code path with custom reduction (works on CPU):
        if src.layout == torch.sparse_csr and not src.is_cuda:
            return torch.sparse.mm(src, other, reduce)

        # Simulate `mean` reduction by dividing by degree:
        if reduce == 'mean':
            if src.layout == torch.sparse_csr:
                ptr = src.crow_indices()
                deg = ptr[1:] - ptr[:-1]
            else:
                assert src.layout == torch.sparse_csc
                deg = scatter(torch.ones_like(src.values()), src.row_indices(),
                              dim=0, dim_size=src.size(0), reduce='sum')

            return torch.sparse.mm(src, other) / deg.view(-1, 1).clamp_(min=1)

        # TODO The `torch.sparse.mm` code path with the `reduce` argument does
        # not yet support CSC :(
        if src.layout == torch.sparse_csc:
            warnings.warn(f"Converting sparse tensor to CSR format for more "
                          f"efficient processing. Consider converting your "
                          f"sparse tensor to CSR format beforehand to avoid "
                          f"repeated conversion (got '{src.layout}')")
            src = src.to_sparse_csr()

        return torch.sparse.mm(src, other, reduce)

    # pragma: no cover
    # PyTorch < 2.0 only supports sparse COO format:
    if reduce == 'sum':
        return torch.sparse.mm(src, other)
    elif reduce == 'mean':
        if src.layout == torch.sparse_csr:
            ptr = src.crow_indices()
            deg = ptr[1:] - ptr[:-1]
        elif src.layout == torch.sparse_csc:
            assert src.layout == torch.sparse_csc
            deg = scatter(torch.ones_like(src.values()), src.row_indices(),
                          dim=0, dim_size=src.size(0), reduce='sum')
        else:
            assert src.layout == torch.sparse_coo
            src = src.coalesce()
            deg = scatter(torch.ones_like(src.values()),
                          src.indices()[0], dim=0, dim_size=src.size(0),
                          reduce='sum')

        return torch.sparse.mm(src, other) / deg.view(-1, 1).clamp_(min=1)

    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"'torch.sparse.Tensor' on device '{src.device}'")

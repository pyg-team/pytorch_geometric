from itertools import product

import pytest
import torch
import torch_scatter
from torch_sparse.tensor import SparseTensor

# from utils import devices, grad_dtypes, reductions


# @pytest.mark.parametrize('dtype,device,reduce',
#                          product(grad_dtypes, devices, reductions))
@pytest.mark.parametrize('dtype,device,reduce',
                         product([torch.float, torch.double], ['cpu'],
                                 ['sum']))
def test_spmm(dtype, device, reduce):
    src = torch.randn((10, 8), dtype=dtype, device=device)
    src[2:4, :] = 0  # Remove multiple rows.
    src[:, 2:4] = 0  # Remove multiple columns.
    src = SparseTensor.from_dense(src).requires_grad_()
    row, col, value = src.coo()

    other = torch.randn((2, 8, 2), dtype=dtype, device=device,
                        requires_grad=True)

    src_col = other.index_select(-2, col) * value.unsqueeze(-1)
    expected = torch_scatter.scatter(src_col, row, dim=-2, reduce=reduce)
    if reduce == 'min':
        expected[expected > 1000] = 0
    if reduce == 'max':
        expected[expected < -1000] = 0

    # grad_out = torch.randn_like(expected)

    # expected.backward(grad_out)
    # expected_grad_value = value.grad
    # value.grad = None
    # expected_grad_other = other.grad
    # other.grad = None

    out_torch_sparse = spmm(src, other, reduce, version="torch_sparse")
    out_torch_spmm = spmm(src, other, reduce, version="torch.spmm")
    # out_torch_spmm.backward(grad_out)
    assert torch.allclose(out_torch_spmm, out_torch_sparse, atol=1e-2)
    # assert torch.allclose(expected_grad_value, value.grad, atol=1e-2)
    # assert torch.allclose(expected_grad_other, other.grad, atol=1e-2)


def spmm(src: SparseTensor, other: torch.Tensor, reduce: str = "sum",
         version: str = "torch_sparse") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other, version)


def spmm_sum(src: SparseTensor, other: torch.Tensor,
             version: str) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage._row
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    if value is not None:
        value = value.to(other.dtype)

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()
    if version == "torch_sparse":
        return torch.ops.torch_sparse.spmm_sum(row, rowptr, col, value, colptr,
                                               csr2csc, other)
    if version == "torch.spmm":
        return torch.spmm_sum(rowptr, col, value, other)

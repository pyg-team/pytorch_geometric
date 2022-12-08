from itertools import product

import pytest
import torch
import torch_scatter
from torch import Tensor
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm as sparse_spmm
from utils import devices, grad_dtypes, reductions

from torch_geometric.utils import spmm


@pytest.mark.parametrize('dtype,device,reduce',
                         product(grad_dtypes, ['cpu'], reductions))
def test_backward_compatibility_spmm(dtype, device, reduce):
    m, n, k = 10, 8, 4
    _src = torch.randn((m, n), dtype=dtype, device=device)
    _src[2:4, :] = 0  # Remove multiple rows.
    _src[:, 2:4] = 0  # Remove multiple columns.

    # sparse src
    src = SparseTensor.from_dense(_src).requires_grad_()
    src_ref = SparseTensor.from_dense(_src).requires_grad_()

    # dense other
    other = torch.randn((n, k), dtype=dtype, device=device, requires_grad=True)
    other_ref = other.clone().detach().requires_grad_()
    other_csr = other.clone().detach().requires_grad_()

    row, col, value = src.coo()
    _, _, value_ref = src_ref.coo()

    # Test with explicit conversion
    csr = src.to_torch_sparse_csr_tensor(dtype=other).requires_grad_()

    # Test against scatter reduce
    src_col = other.index_select(-2, col) * value.unsqueeze(-1)
    expected = torch_scatter.scatter(src_col, row, dim=-2, reduce=reduce)
    grad_out = torch.randn_like(expected)

    # Test against implementation in pytorch_sparse
    out_ref = compare_spmm(src_ref, other_ref, reduce, version="torch_sparse")
    out_ref.backward(grad_out)
    out = compare_spmm(src, other, reduce, version="torch.spmm")
    out.backward(grad_out)

    out_csr = compare_spmm(csr, other_csr, reduce, version="torch.spmm")
    out_csr.backward(grad_out)

    assert torch.allclose(out_ref, out)
    assert torch.allclose(value_ref.grad, csr.grad.values(),
                          atol=10 - 4)  #? This passes
    assert torch.allclose(other_ref.grad, other.grad)


def compare_spmm(src: SparseTensor, other: torch.Tensor, reduce: str,
                 version: str) -> torch.Tensor:

    if version == "torch_sparse":
        return sparse_spmm(src, other, reduce)
    if version == "torch.spmm":
        return spmm(src, other, reduce)


@pytest.mark.parametrize('dtype,reduce', product(grad_dtypes, reductions))
def test_spmm(dtype, reduce):
    @torch.jit.script
    def jit_torch_sparse(src: SparseTensor, other: Tensor) -> Tensor:
        return spmm(src, other)

    @torch.jit.script
    def jit_torch(src: Tensor, other: Tensor) -> Tensor:
        return spmm(src, other, reduce='sum')

    if reduce in ['mean', 'min', 'max']:
        src = torch.tensor([[1, 1], [0, 0]], dtype=dtype)
        other = torch.tensor([[1, -1], [99, -99]], dtype=dtype)
        out = spmm(SparseTensor.from_dense(src), other, reduce)
        if reduce == 'mean':
            assert out.sum() == 0
        elif reduce == 'max':
            assert out.sum() > 0
        elif reduce == 'max':
            assert out.sum() < 0

    elif reduce in ['sum', 'add']:
        src = torch.randn(5, 4, dtype=dtype)
        other = torch.randn(4, 8, dtype=dtype)

        out1 = src @ other
        src = SparseTensor.from_dense(src)
        out2 = jit_torch_sparse(src, other, reduce=reduce)
        src = src.to_torch_sparse_csr_tensor(dtype=other)
        out3 = jit_torch(src, other, reduce=reduce)
        assert out1.size() == (5, 8)
        assert torch.allclose(out1, out2)
        assert torch.allclose(out1, out3)

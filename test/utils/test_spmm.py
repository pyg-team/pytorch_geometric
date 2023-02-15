import pytest
import torch
from packaging import version
from torch import Tensor

from torch_geometric.typing import SparseTensor
from torch_geometric.utils import spmm


def test_spmm():
    src = torch.randn(5, 4)
    other = torch.randn(4, 8)

    out1 = src @ other
    out2 = spmm(SparseTensor.from_dense(src), other, reduce='sum')
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2, atol=1e-8)

    for reduce in ['sum', 'mean', 'min', 'max']:
        out = spmm(SparseTensor.from_dense(src), other, reduce)
        assert out.size() == (5, 8)

    if version.parse(torch.__version__).base_version >= version.parse(
            '2.0.0').base_version:
        src = src.to_sparse_csr()
        out3 = spmm(src, other, reduce='sum')
        assert torch.allclose(out1, out3, atol=1e-8)

        for reduce in ['sum', 'mean', 'min', 'max']:
            out = spmm(src, other, reduce)
            assert out.size() == (5, 8)


def test_spmm_jit():
    @torch.jit.script
    def jit_torch_sparse(src: SparseTensor, other: Tensor) -> Tensor:
        return spmm(src, other, reduce='sum')

    src = torch.randn(5, 4)
    other = torch.randn(4, 8)

    out1 = src @ other
    out2 = jit_torch_sparse(SparseTensor.from_dense(src), other)
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2)

    if version.parse(torch.__version__).base_version >= version.parse(
            '2.0.0').base_version:

        @torch.jit.script
        def jit_torch(src: Tensor, other: Tensor) -> Tensor:
            return spmm(src, other, reduce='sum')

        out3 = jit_torch(src.to_sparse_csr(), other)
        assert torch.allclose(out1, out3)

import pytest
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.utils import spmm


def test_spmm():
    src = torch.randn(5, 4)
    other = torch.randn(4, 8)

    out1 = src @ other
    out2 = spmm(src.to_sparse(), other, reduce='sum')
    out3 = spmm(SparseTensor.from_dense(src), other, reduce='sum')
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, out3)

    for reduce in ['mean', 'min', 'max']:
        out = spmm(SparseTensor.from_dense(src), other, reduce)
        assert out.size() == (5, 8)

        with pytest.raises(ValueError, match="not supported"):
            spmm(src.to_sparse(), other, reduce)


def test_spmm():
    @torch.jit.script
    def jit_torch_sparse(src: SparseTensor, other: Tensor) -> Tensor:
        return spmm(src, other)

    @torch.jit.script
    def jit_torch(src: Tensor, other: Tensor) -> Tensor:
        return spmm(src, other, reduce='sum')

    src = torch.randn(5, 4)
    other = torch.randn(4, 8)

    out1 = src @ other
    out2 = jit_torch_sparse(SparseTensor.from_dense(src), other)
    out3 = jit_torch(src.to_sparse(), other)
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, out3)

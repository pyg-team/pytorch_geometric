import pytest
import torch
from torch import Tensor

from torch_geometric.testing import withCUDA
from torch_geometric.typing import WITH_PT2, SparseTensor
from torch_geometric.utils import spmm


@withCUDA
def test_spmm_basic(device):
    src = torch.randn(5, 4, device=device)
    other = torch.randn(4, 8, device=device)

    out1 = src @ other
    out2 = spmm(src.to_sparse(), other, reduce='sum')
    out3 = spmm(SparseTensor.from_dense(src), other, reduce='sum')
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, out3)


@withCUDA
@pytest.mark.parametrize('reduce', ['mean', 'min', 'max'])
def test_spmm_reduce(device, reduce):
    src = torch.randn(5, 4, device=device)
    other = torch.randn(4, 8, device=device)

    if WITH_PT2:
        if src.is_cuda:
            with pytest.raises(NotImplementedError, match="doesn't exist"):
                spmm(src.to_sparse(), other, reduce=reduce)
        else:
            out1 = spmm(src.to_sparse(), other, reduce=reduce)
            out2 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
            assert torch.allclose(out1, out2)
    else:
        with pytest.raises(ValueError, match="not supported"):
            spmm(src.to_sparse(), other, reduce)


def test_spmm_jit():
    @torch.jit.script
    def jit_torch_sparse(src: SparseTensor, other: Tensor) -> Tensor:
        return spmm(src, other, reduce='sum')

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

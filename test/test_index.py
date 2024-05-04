import pytest
import torch

from torch_geometric import Index
from torch_geometric.testing import withCUDA
from torch_geometric.typing import INDEX_DTYPES

DTYPES = [pytest.param(dtype, id=str(dtype)[6:]) for dtype in INDEX_DTYPES]


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_basic(dtype, device):
    kwargs = dict(dtype=dtype, device=device, dim_size=3)
    index = Index([0, 1, 1, 2], **kwargs)
    index.validate()
    assert isinstance(index, Index)

    assert str(index).startswith('Index([0, 1, 1, 2], ')
    assert 'dim_size=3' in str(index)
    assert (f"device='{device}'" in str(index)) == index.is_cuda
    assert (f'dtype={dtype}' in str(index)) == (dtype != torch.long)

    assert index.dtype == dtype
    assert index.device == device
    assert index.dim_size == 3
    assert not index.is_sorted

    out = index.as_tensor()
    assert not isinstance(out, Index)
    assert out.dtype == dtype
    assert out.device == device

    out = index * 1
    assert not isinstance(out, Index)
    assert out.dtype == dtype
    assert out.device == device


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_identity(dtype, device):
    kwargs = dict(dtype=dtype, device=device, dim_size=3, is_sorted=True)
    index = Index([0, 1, 1, 2], **kwargs)

    out = Index(index)
    assert not isinstance(out.as_tensor(), Index)
    assert out.data_ptr() == index.data_ptr()
    assert out.dtype == index.dtype
    assert out.device == index.device
    assert out.dim_size == index.dim_size
    assert out.is_sorted == index.is_sorted

    out = Index(index, dim_size=4, is_sorted=False)
    assert out.dim_size == 4
    assert out.is_sorted == index.is_sorted

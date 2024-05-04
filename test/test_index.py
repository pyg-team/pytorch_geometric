import pytest
import torch
from torch import tensor

import torch_geometric.typing
from torch_geometric import Index
from torch_geometric.testing import onlyCUDA, withCUDA
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


def test_validate():
    with pytest.raises(ValueError, match="unsupported data type"):
        Index([0.0, 1.0])
    with pytest.raises(ValueError, match="needs to be one-dimensional"):
        Index([[0], [1]])
    with pytest.raises(TypeError, match="invalid combination of arguments"):
        Index(torch.tensor([0, 1]), torch.long)
    with pytest.raises(TypeError, match="invalid keyword arguments"):
        Index(torch.tensor([0, 1]), dtype=torch.long)
    with pytest.raises(ValueError, match="contains negative indices"):
        Index([-1, 0]).validate()
    with pytest.raises(ValueError, match="than its registered size"):
        Index([0, 10], dim_size=2).validate()
    with pytest.raises(ValueError, match="not sorted"):
        Index([1, 0], is_sorted=True).validate()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_fill_cache_(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], is_sorted=True, **kwargs)
    index.validate().fill_cache_()
    assert index.dim_size == 3
    assert index._indptr.dtype == dtype
    assert index._indptr.equal(tensor([0, 1, 3, 4], device=device))

    index = Index([1, 0, 2, 1], **kwargs)
    index.validate().fill_cache_()
    assert index.dim_size == 3
    assert index._indptr is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_dim_resize(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], is_sorted=True, **kwargs).fill_cache_()

    assert index.dim_size == 3
    assert index._indptr.equal(tensor([0, 1, 3, 4], device=device))

    out = index.dim_resize_(4)
    assert out.dim_size == 4
    assert out._indptr.equal(tensor([0, 1, 3, 4, 4], device=device))

    out = index.dim_resize_(3)
    assert out.dim_size == 3
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))

    out = index.dim_resize_(None)
    assert out.dim_size is None
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_clone(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], is_sorted=True, dim_size=3, **kwargs)

    out = index.clone()
    assert isinstance(out, Index)
    assert out.dtype == dtype
    assert out.device == device
    assert out.dim_size == 3
    assert out.is_sorted

    out = torch.clone(index)
    assert isinstance(out, Index)
    assert out.dtype == dtype
    assert out.device == device
    assert out.dim_size == 3
    assert out.is_sorted


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_to_function(dtype, device):
    kwargs = dict(dtype=dtype)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index.fill_cache_()

    index = index.to(device)
    assert isinstance(index, Index)
    assert index.device == device
    assert index._indptr.dtype == dtype
    assert index._indptr.device == device

    out = index.cpu()
    assert isinstance(out, Index)
    assert out.device == torch.device('cpu')

    out = index.to(torch.int)
    assert out.dtype == torch.int
    if torch_geometric.typing.WITH_PT20:
        assert isinstance(out, Index)
        assert out._indptr.dtype == torch.int
    else:
        assert not isinstance(out, Index)

    out = index.to(torch.float)
    assert not isinstance(out, Index)
    assert out.dtype == torch.float

    out = index.long()
    assert isinstance(out, Index)
    assert out.dtype == torch.int64

    out = index.int()
    assert out.dtype == torch.int
    if torch_geometric.typing.WITH_PT20:
        assert isinstance(out, Index)
    else:
        assert not isinstance(out, Index)


@onlyCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_cpu_cuda(dtype):
    kwargs = dict(dtype=dtype)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    assert index.is_cpu

    out = index.cuda()
    assert isinstance(out, Index)
    assert out.is_cuda

    out = out.cpu()
    assert isinstance(out, Index)
    assert out.is_cpu


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_share_memory(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index.fill_cache_()

    out = index.share_memory_()
    assert isinstance(out, Index)
    assert out.is_shared()
    assert out._data.is_shared()
    assert out._indptr.is_shared()
    assert out.data_ptr() == index.data_ptr()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_contiguous(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)

    assert index.is_contiguous
    out = index.contiguous()
    assert isinstance(out, Index)
    assert out.is_contiguous


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sort(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([1, 0, 2, 1], dim_size=3, **kwargs)

    index, _ = index.sort()
    assert isinstance(index, Index)
    assert index.equal(tensor([0, 1, 1, 2], device=device))
    assert index.dim_size == 3
    assert index.is_sorted

    out, perm = index.sort()
    assert isinstance(out, Index)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert perm.equal(tensor([0, 1, 2, 3], device=device))
    assert out.dim_size == 3

    index, _ = index.sort(descending=True)
    assert isinstance(index, Index)
    assert index.equal(tensor([2, 1, 1, 0], device=device))
    assert index.dim_size == 3
    assert not index.is_sorted


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sort_stable(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([1, 0, 2, 1], dim_size=3, **kwargs)

    index, perm = index.sort(stable=True)
    assert isinstance(index, Index)
    assert index.equal(tensor([0, 1, 1, 2], device=device))
    assert perm.equal(tensor([1, 0, 3, 2], device=device))
    assert index.dim_size == 3
    assert index.is_sorted

    out, perm = index.sort(stable=True)
    assert isinstance(out, Index)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert perm.equal(tensor([0, 1, 2, 3], device=device))
    assert out.dim_size == 3

    index, perm = index.sort(descending=True, stable=True)
    assert isinstance(index, Index)
    assert index.equal(tensor([2, 1, 1, 0], device=device))
    assert perm.equal(tensor([3, 1, 2, 0], device=device))
    assert index.dim_size == 3
    assert not index.is_sorted

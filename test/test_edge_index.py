import os.path as osp
import warnings
from typing import List, Optional

import pytest
import torch
from torch import Tensor, tensor

import torch_geometric
from torch_geometric import EdgeIndex, Index
from torch_geometric.edge_index import (
    ReduceType,
    SortReturnType,
    _scatter_spmm,
    _torch_sparse_spmm,
    _TorchSPMM,
    set_tuple_item,
)
from torch_geometric.io import fs
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    onlyCUDA,
    onlyLinux,
    withCUDA,
    withoutExtensions,
    withPackage,
)
from torch_geometric.typing import INDEX_DTYPES, SparseTensor
from torch_geometric.utils import scatter

DTYPES = [pytest.param(dtype, id=str(dtype)[6:]) for dtype in INDEX_DTYPES]
IS_UNDIRECTED = [
    pytest.param(False, id='directed'),
    pytest.param(True, id='undirected'),
]
TRANSPOSE = [
    pytest.param(False, id=''),
    pytest.param(True, id='transpose'),
]


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_basic(dtype, device):
    kwargs = dict(dtype=dtype, device=device, sparse_size=(3, 3))
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    adj.validate()
    assert isinstance(adj, EdgeIndex)

    assert str(adj).startswith('EdgeIndex([[0, 1, 1, 2],\n'
                               '           [1, 0, 2, 1]], ')
    assert 'sparse_size=(3, 3), nnz=4' in str(adj)
    assert (f"device='{device}'" in str(adj)) == adj.is_cuda
    assert (f'dtype={dtype}' in str(adj)) == (dtype != torch.long)

    assert adj.dtype == dtype
    assert adj.device == device
    assert adj.sparse_size() == (3, 3)
    assert adj.sparse_size(0) == 3
    assert adj.sparse_size(-1) == 3

    assert adj.sort_order is None
    assert not adj.is_sorted
    assert not adj.is_sorted_by_row
    assert not adj.is_sorted_by_col

    assert not adj.is_undirected

    out = adj.as_tensor()
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert out.device == device

    out = adj * 1
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert out.device == device


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_identity(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3), **kwargs)

    out = EdgeIndex(adj)
    assert not isinstance(out.as_tensor(), EdgeIndex)
    assert out.data_ptr() == adj.data_ptr()
    assert out.dtype == adj.dtype
    assert out.device == adj.device
    assert out.sparse_size() == adj.sparse_size()
    assert out.sort_order == adj.sort_order
    assert out.is_undirected == adj.is_undirected

    out = EdgeIndex(adj, sparse_size=(4, 4), sort_order='row')
    assert out.sparse_size() == (4, 4)
    assert out.sort_order == 'row'


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sparse_tensor(dtype, device):
    kwargs = dict(dtype=dtype, device=device, is_undirected=True)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    out = EdgeIndex(adj.to_sparse_coo())
    assert out.equal(adj)
    assert out.sort_order == 'row'
    assert out.sparse_size() == (3, 3)
    assert out._indptr is None

    out = EdgeIndex(adj.to_sparse_csr())
    assert out.equal(adj)
    assert out.sort_order == 'row'
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))

    out = EdgeIndex(adj.to_sparse_csc())
    assert out.equal(adj.sort_by('col')[0])
    assert out.sort_order == 'col'
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))


def test_set_tuple_item():
    tmp = (0, 1, 2)
    assert set_tuple_item(tmp, 0, 3) == (3, 1, 2)
    assert set_tuple_item(tmp, 1, 3) == (0, 3, 2)
    assert set_tuple_item(tmp, 2, 3) == (0, 1, 3)
    with pytest.raises(IndexError, match="tuple index out of range"):
        set_tuple_item(tmp, 3, 3)
    assert set_tuple_item(tmp, -1, 3) == (0, 1, 3)
    assert set_tuple_item(tmp, -2, 3) == (0, 3, 2)
    assert set_tuple_item(tmp, -3, 3) == (3, 1, 2)
    with pytest.raises(IndexError, match="tuple index out of range"):
        set_tuple_item(tmp, -4, 3)


def test_validate():
    with pytest.raises(TypeError, match="tensors of a single element"):
        EdgeIndex([torch.tensor([0, 1]), torch.tensor([1, 0])])
    with pytest.raises(ValueError, match="unsupported data type"):
        EdgeIndex([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="needs to be two-dimensional"):
        EdgeIndex([[[0], [1]], [[1], [0]]])
    with pytest.raises(ValueError, match="needs to have a shape of"):
        EdgeIndex([[0, 1], [1, 0], [1, 1]])
    with pytest.raises(ValueError, match="received a non-symmetric size"):
        EdgeIndex([[0, 1], [1, 0]], is_undirected=True, sparse_size=(2, 3))
    with pytest.raises(TypeError, match="invalid combination of arguments"):
        EdgeIndex(tensor([[0, 1], [1, 0]]), torch.long)
    with pytest.raises(TypeError, match="invalid keyword arguments"):
        EdgeIndex(tensor([[0, 1], [1, 0]]), dtype=torch.long)
    with pytest.raises(ValueError, match="contains negative indices"):
        EdgeIndex([[-1, 0], [0, 1]]).validate()
    with pytest.raises(ValueError, match="than its number of rows"):
        EdgeIndex([[0, 10], [1, 0]], sparse_size=(2, 2)).validate()
    with pytest.raises(ValueError, match="than its number of columns"):
        EdgeIndex([[0, 1], [10, 0]], sparse_size=(2, 2)).validate()
    with pytest.raises(ValueError, match="not sorted by row indices"):
        EdgeIndex([[1, 0], [0, 1]], sort_order='row').validate()
    with pytest.raises(ValueError, match="not sorted by column indices"):
        EdgeIndex([[0, 1], [1, 0]], sort_order='col').validate()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_undirected(dtype, device):
    kwargs = dict(dtype=dtype, device=device, is_undirected=True)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    assert isinstance(adj, EdgeIndex)
    assert adj.is_undirected

    assert adj.sparse_size() == (None, None)
    adj.get_num_rows()
    assert adj.sparse_size() == (3, 3)
    adj.validate()

    adj = EdgeIndex([[0, 1], [1, 0]], sparse_size=(3, None), **kwargs)
    assert adj.sparse_size() == (3, 3)
    adj.validate()

    adj = EdgeIndex([[0, 1], [1, 0]], sparse_size=(None, 3), **kwargs)
    assert adj.sparse_size() == (3, 3)
    adj.validate()

    with pytest.raises(ValueError, match="'EdgeIndex' is not undirected"):
        EdgeIndex([[0, 1, 1, 2], [0, 0, 1, 1]], **kwargs).validate()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_fill_cache_(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    adj.validate().fill_cache_()
    assert adj.sparse_size() == (3, 3)
    assert adj._indptr.dtype == dtype
    assert adj._indptr.equal(tensor([0, 1, 3, 4], device=device))
    assert adj._T_perm.dtype == dtype
    assert (adj._T_perm.equal(tensor([1, 0, 3, 2], device=device))
            or adj._T_perm.equal(tensor([1, 3, 0, 2], device=device)))
    assert adj._T_index[0].dtype == dtype
    assert (adj._T_index[0].equal(tensor([1, 0, 2, 1], device=device))
            or adj._T_index[0].equal(tensor([1, 2, 0, 1], device=device)))
    assert adj._T_index[1].dtype == dtype
    assert adj._T_index[1].equal(tensor([0, 1, 1, 2], device=device))
    if is_undirected:
        assert adj._T_indptr is None
    else:
        assert adj._T_indptr.dtype == dtype
        assert adj._T_indptr.equal(tensor([0, 1, 3, 4], device=device))

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order='col', **kwargs)
    adj.validate().fill_cache_()
    assert adj.sparse_size() == (3, 3)
    assert adj._indptr.dtype == dtype
    assert adj._indptr.equal(tensor([0, 1, 3, 4], device=device))
    assert (adj._T_perm.equal(tensor([1, 0, 3, 2], device=device))
            or adj._T_perm.equal(tensor([1, 3, 0, 2], device=device)))
    assert adj._T_index[0].dtype == dtype
    assert adj._T_index[0].equal(tensor([0, 1, 1, 2], device=device))
    assert adj._T_index[1].dtype == dtype
    assert (adj._T_index[1].equal(tensor([1, 0, 2, 1], device=device))
            or adj._T_index[1].equal(tensor([1, 2, 0, 1], device=device)))
    if is_undirected:
        assert adj._T_indptr is None
    else:
        assert adj._T_indptr.dtype == dtype
        assert adj._T_indptr.equal(tensor([0, 1, 3, 4], device=device))


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_clone(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    out = adj.clone()
    assert isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert out.device == device
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected

    out = torch.clone(adj)
    assert isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert out.device == device
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_to_function(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    adj.fill_cache_()

    adj = adj.to(device)
    assert isinstance(adj, EdgeIndex)
    assert adj.device == device
    assert adj._indptr.dtype == dtype
    assert adj._indptr.device == device
    assert adj._T_perm.dtype == dtype
    assert adj._T_perm.device == device

    out = adj.cpu()
    assert isinstance(out, EdgeIndex)
    assert out.device == torch.device('cpu')

    out = adj.to(torch.int)
    assert out.dtype == torch.int
    if torch_geometric.typing.WITH_PT20:
        assert isinstance(out, EdgeIndex)
        assert out._indptr.dtype == torch.int
        assert out._T_perm.dtype == torch.int
    else:
        assert not isinstance(out, EdgeIndex)

    out = adj.to(torch.float)
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == torch.float

    out = adj.long()
    assert isinstance(out, EdgeIndex)
    assert out.dtype == torch.int64

    out = adj.int()
    assert out.dtype == torch.int
    if torch_geometric.typing.WITH_PT20:
        assert isinstance(out, EdgeIndex)
    else:
        assert not isinstance(out, EdgeIndex)


@onlyCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_cpu_cuda(dtype):
    kwargs = dict(dtype=dtype)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    assert adj.is_cpu

    out = adj.cuda()
    assert isinstance(out, EdgeIndex)
    assert out.is_cuda

    out = out.cpu()
    assert isinstance(out, EdgeIndex)
    assert out.is_cpu


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_share_memory(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    adj.fill_cache_()

    out = adj.share_memory_()
    assert isinstance(out, EdgeIndex)
    assert out.is_shared()
    assert out._data.is_shared()
    assert out._indptr.is_shared()
    assert out.data_ptr() == adj.data_ptr()


@onlyCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_pin_memory(dtype):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=dtype)
    assert not adj.is_pinned()
    out = adj.pin_memory()
    assert out.is_pinned()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_contiguous(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    data = tensor([[0, 1], [1, 0], [1, 2], [2, 1]], **kwargs).t()

    with pytest.raises(ValueError, match="needs to be contiguous"):
        EdgeIndex(data)

    adj = EdgeIndex(data.contiguous()).contiguous()
    assert isinstance(adj, EdgeIndex)
    assert adj.is_contiguous()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_sort_by(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    out = adj.sort_by('row')
    assert isinstance(out, SortReturnType)
    assert isinstance(out.values, EdgeIndex)
    assert not isinstance(out.indices, EdgeIndex)
    assert out.values.equal(adj)
    assert out.indices is None

    adj = EdgeIndex([[0, 1, 2, 1], [1, 0, 1, 2]], **kwargs)
    out = adj.sort_by('row')
    assert isinstance(out, SortReturnType)
    assert isinstance(out.values, EdgeIndex)
    assert not isinstance(out.indices, EdgeIndex)
    assert out.values[0].equal(tensor([0, 1, 1, 2], device=device))
    assert (out.values[1].equal(tensor([1, 0, 2, 1], device=device))
            or out.values[1].equal(tensor([1, 2, 0, 1], device=device)))
    assert (out.indices.equal(tensor([0, 1, 3, 2], device=device))
            or out.indices.equal(tensor([0, 3, 1, 2], device=device)))

    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    out, perm = adj.sort_by('col')
    assert adj._T_perm is not None  # Check caches.
    assert adj._T_index[0] is not None and adj._T_index[1] is not None
    assert (out[0].equal(tensor([1, 0, 2, 1], device=device))
            or out[0].equal(tensor([1, 2, 0, 1], device=device)))
    assert out[1].equal(tensor([0, 1, 1, 2], device=device))
    assert (perm.equal(tensor([1, 0, 3, 2], device=device))
            or perm.equal(tensor([1, 3, 0, 2], device=device)))
    assert out._T_perm is None
    assert out._T_index[0] is None and out._T_index[1] is None

    out, perm = out.sort_by('row')
    assert out[0].equal(tensor([0, 1, 1, 2], device=device))
    assert (out[1].equal(tensor([1, 0, 2, 1], device=device))
            or out[1].equal(tensor([1, 2, 0, 1], device=device)))
    assert (perm.equal(tensor([1, 0, 3, 2], device=device))
            or perm.equal(tensor([2, 3, 0, 1], device=device)))
    assert out._T_perm is None
    assert out._T_index[0] is None and out._T_index[1] is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_cat(dtype, device, is_undirected):
    args = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj1 = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3), **args)
    adj2 = EdgeIndex([[1, 2, 2, 3], [2, 1, 3, 2]], sparse_size=(4, 4), **args)
    adj3 = EdgeIndex([[1, 2, 2, 3], [2, 1, 3, 2]], dtype=dtype, device=device)

    out = torch.cat([adj1, adj2], dim=1)
    assert out.size() == (2, 8)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size() == (4, 4)
    assert not out.is_sorted
    assert out.is_undirected == is_undirected

    assert out._cat_metadata.nnz == [4, 4]
    assert out._cat_metadata.sparse_size == [(3, 3), (4, 4)]
    assert out._cat_metadata.sort_order == [None, None]
    assert out._cat_metadata.is_undirected == [is_undirected, is_undirected]

    out = torch.cat([adj1, adj2, adj3], dim=1)
    assert out.size() == (2, 12)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size() == (None, None)
    assert not out.is_sorted
    assert not out.is_undirected

    out = torch.cat([adj1, adj2], dim=0)
    assert out.size() == (4, 4)
    assert not isinstance(out, EdgeIndex)

    inplace = torch.empty(2, 8, dtype=dtype, device=device)
    out = torch.cat([adj1, adj2], dim=1, out=inplace)
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, EdgeIndex)
    assert not isinstance(inplace, EdgeIndex)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_flip(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    adj.fill_cache_()

    out = adj.flip(0)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[1, 0, 2, 1], [0, 1, 1, 2]], device=device))
    assert out.is_sorted_by_col
    assert out.is_undirected == is_undirected
    assert out._T_indptr.equal(tensor([0, 1, 3, 4], device=device))

    out = adj.flip([0, 1])
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[1, 2, 0, 1], [2, 1, 1, 0]], device=device))
    assert not out.is_sorted
    assert out.is_undirected == is_undirected
    assert out._T_indptr is None

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order='col', **kwargs)
    out = adj.flip(0)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device))
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_index_select(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    index = tensor([1, 3], device=device)
    out = adj.index_select(1, index)
    assert out.equal(tensor([[1, 2], [0, 1]], device=device))
    assert isinstance(out, EdgeIndex)
    assert not out.is_sorted
    assert not out.is_undirected

    index = tensor([0], device=device)
    out = adj.index_select(0, index)
    assert out.equal(tensor([[0, 1, 1, 2]], device=device))
    assert not isinstance(out, EdgeIndex)

    index = tensor([1, 3], device=device)
    inplace = torch.empty(2, 2, dtype=dtype, device=device)
    out = torch.index_select(adj, 1, index, out=inplace)
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, EdgeIndex)
    assert not isinstance(inplace, EdgeIndex)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_narrow(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    out = adj.narrow(dim=1, start=1, length=2)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[1, 1], [0, 2]], device=device))
    assert out.is_sorted_by_row
    assert not out.is_undirected

    out = adj.narrow(dim=0, start=0, length=1)
    assert not isinstance(out, EdgeIndex)
    assert out.equal(tensor([[0, 1, 1, 2]], device=device))


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_getitem(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    out = adj[:, tensor([False, True, False, True], device=device)]
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[1, 2], [0, 1]], device=device))
    assert out.is_sorted_by_row
    assert not out.is_undirected

    out = adj[..., tensor([1, 3], device=device)]
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[1, 2], [0, 1]], device=device))
    assert not out.is_sorted
    assert not out.is_undirected

    out = adj[..., 1::2]
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[1, 2], [0, 1]], device=device))
    assert out.is_sorted_by_row
    assert not out.is_undirected

    out = adj[...]
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device))
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected

    out = adj[None]
    assert not isinstance(out, EdgeIndex)
    assert out.equal(tensor([[[0, 1, 1, 2], [1, 0, 2, 1]]], device=device))

    out = adj[0, 0]
    assert not isinstance(out, EdgeIndex)
    assert out.equal(tensor(0, device=device))

    out = adj[:, 0]
    assert not isinstance(out, EdgeIndex)

    out = adj[tensor([0], device=device)]
    assert not isinstance(out, EdgeIndex)

    out = adj[tensor([0], device=device), tensor([0], device=device)]
    assert not isinstance(out, EdgeIndex)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_select(dtype, device):
    kwargs = dict(dtype=dtype, device=device)

    adj = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sort_order='row',
        sparse_size=(4, 5),
        **kwargs,
    ).fill_cache_()

    out = adj[0]
    assert isinstance(out, Index)
    assert out.equal(tensor([0, 1, 1, 2], device=device))
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr.equal(tensor([0, 1, 3, 4, 4], device=device))

    out = adj[-1]
    assert isinstance(out, Index)
    assert out.equal(tensor([1, 0, 2, 1], device=device))
    assert out.dim_size == 5
    assert not out.is_sorted
    assert out._indptr is None

    out = adj[-2, 2:4]
    assert isinstance(out, Index)
    assert out.equal(tensor([1, 2], device=device))
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr is None

    adj = EdgeIndex(
        [[1, 0, 2, 1], [0, 1, 1, 2]],
        sort_order='col',
        sparse_size=(5, 4),
        **kwargs,
    ).fill_cache_()

    out = adj[1]
    assert isinstance(out, Index)
    assert out.equal(tensor([0, 1, 1, 2], device=device))
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr.equal(tensor([0, 1, 3, 4, 4], device=device))

    out = adj[-2]
    assert isinstance(out, Index)
    assert out.equal(tensor([1, 0, 2, 1], device=device))
    assert out.dim_size == 5
    assert not out.is_sorted
    assert out._indptr is None

    out = adj[-1, 2:4]
    assert isinstance(out, Index)
    assert out.equal(tensor([1, 2], device=device))
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_unbind(dtype, device):
    kwargs = dict(dtype=dtype, device=device)

    adj = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sort_order='row',
        sparse_size=(4, 5),
        **kwargs,
    ).fill_cache_()

    row, col = adj

    assert isinstance(row, Index)
    assert row.equal(tensor([0, 1, 1, 2], device=device))
    assert row.dim_size == 4
    assert row.is_sorted
    assert row._indptr.equal(tensor([0, 1, 3, 4, 4], device=device))

    assert isinstance(col, Index)
    assert col.equal(tensor([1, 0, 2, 1], device=device))
    assert col.dim_size == 5
    assert not col.is_sorted
    assert col._indptr is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('value_dtype', [None, torch.double])
def test_to_dense(dtype, device, value_dtype):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], **kwargs)

    out = adj.to_dense(dtype=value_dtype)
    assert isinstance(out, Tensor)
    assert out.size() == (3, 3)
    expected = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    assert out.equal(tensor(expected, dtype=value_dtype, device=device))

    value = torch.arange(1, 5, dtype=value_dtype or torch.float, device=device)
    out = adj.to_dense(value)
    assert isinstance(out, Tensor)
    assert out.size() == (3, 3)
    expected = [[0.0, 2.0, 0.0], [1.0, 0.0, 4.0], [0.0, 3.0, 0.0]]
    assert out.equal(tensor(expected, dtype=value_dtype, device=device))

    value = torch.arange(1, 5, dtype=value_dtype or torch.float, device=device)
    out = adj.to_dense(value.view(-1, 1))
    assert isinstance(out, Tensor)
    assert out.size() == (3, 3, 1)
    expected = [
        [[0.0], [2.0], [0.0]],
        [[1.0], [0.0], [4.0]],
        [[0.0], [3.0], [0.0]],
    ]
    assert out.equal(tensor(expected, dtype=value_dtype, device=device))


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_to_sparse_coo(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], **kwargs)

    if torch_geometric.typing.WITH_PT20:
        with pytest.raises(ValueError, match="Unexpected tensor layout"):
            adj.to_sparse(layout='int64')

    if torch_geometric.typing.WITH_PT20:
        out = adj.to_sparse(layout=torch.sparse_coo)
    else:
        out = adj.to_sparse()
    assert isinstance(out, Tensor)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.layout == torch.sparse_coo
    assert out.size() == (3, 3)
    assert adj.equal(out._indices())
    assert not out.is_coalesced()

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], **kwargs)
    out = adj.to_sparse_coo()
    assert isinstance(out, Tensor)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.layout == torch.sparse_coo
    assert out.size() == (3, 3)
    assert adj.equal(out._indices())
    assert not out.is_coalesced()

    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    out = adj.to_sparse_coo()
    assert isinstance(out, Tensor)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.layout == torch.sparse_coo
    assert out.size() == (3, 3)
    assert adj.equal(out._indices())
    assert out.is_coalesced()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_to_sparse_csr(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    with pytest.raises(ValueError, match="not sorted"):
        EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs).to_sparse_csr()

    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    if torch_geometric.typing.WITH_PT20:
        out = adj.to_sparse(layout=torch.sparse_csr)
    else:
        out = adj.to_sparse_csr()
    assert isinstance(out, Tensor)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.layout == torch.sparse_csr
    assert out.size() == (3, 3)
    assert adj._indptr.equal(out.crow_indices())
    assert adj[1].equal(out.col_indices())


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_to_sparse_csc(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    with pytest.raises(ValueError, match="not sorted"):
        EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs).to_sparse_csc()

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order='col', **kwargs)
    if torch_geometric.typing.WITH_PT20:
        out = adj.to_sparse(layout=torch.sparse_csc)
    else:
        out = adj.to_sparse_csc()
    assert isinstance(out, Tensor)
    assert out.dtype == torch.float
    assert out.device == device
    assert out.layout == torch.sparse_csc
    assert out.size() == (3, 3)
    assert adj._indptr.equal(out.ccol_indices())
    assert adj[0].equal(out.row_indices())


@withCUDA
@withPackage('torch_sparse')
def test_to_sparse_tensor(device):
    kwargs = dict(device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    out = adj.to_sparse_tensor()
    assert isinstance(out, SparseTensor)
    assert out.sizes() == [3, 3]
    row, col, _ = out.coo()
    assert row.equal(adj[0])
    assert col.equal(adj[1])


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_add(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3), **kwargs)

    out = torch.add(adj, 2, alpha=2)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[4, 5, 5, 6], [5, 4, 6, 5]], device=device))
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (7, 7)

    out = adj + tensor([2], dtype=dtype, device=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[2, 3, 3, 4], [3, 2, 4, 3]], device=device))
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj + tensor([[2], [1]], dtype=dtype, device=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[2, 3, 3, 4], [2, 1, 3, 2]], device=device))
    assert not out.is_undirected
    assert out.sparse_size() == (5, 4)

    out = adj + tensor([[2], [2]], dtype=dtype, device=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[2, 3, 3, 4], [3, 2, 4, 3]], device=device))
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj.add(adj)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[0, 2, 2, 4], [2, 0, 4, 2]], device=device))
    assert not out.is_undirected
    assert out.sparse_size() == (6, 6)

    adj += 2
    assert isinstance(adj, EdgeIndex)
    assert adj.equal(tensor([[2, 3, 3, 4], [3, 2, 4, 3]], device=device))
    assert adj.is_undirected == is_undirected
    assert adj.sparse_size() == (5, 5)

    with pytest.raises(RuntimeError, match="can't be cast"):
        adj += 2.5


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_sub(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[4, 5, 5, 6], [5, 4, 6, 5]], sparse_size=(7, 7), **kwargs)

    out = torch.sub(adj, 2, alpha=2)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device))
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (3, 3)

    out = adj - tensor([2], dtype=dtype, device=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[2, 3, 3, 4], [3, 2, 4, 3]], device=device))
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj - tensor([[2], [1]], dtype=dtype, device=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[2, 3, 3, 4], [4, 3, 5, 4]], device=device))
    assert not out.is_undirected
    assert out.sparse_size() == (5, 6)

    out = adj - tensor([[2], [2]], dtype=dtype, device=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[2, 3, 3, 4], [3, 2, 4, 3]], device=device))
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj.sub(adj)
    assert isinstance(out, EdgeIndex)
    assert out.equal(tensor([[0, 0, 0, 0], [0, 0, 0, 0]], device=device))
    assert not out.is_undirected
    assert out.sparse_size() == (None, None)

    adj -= 2
    assert isinstance(adj, EdgeIndex)
    assert adj.equal(tensor([[2, 3, 3, 4], [3, 2, 4, 3]], device=device))
    assert adj.is_undirected == is_undirected
    assert adj.sparse_size() == (5, 5)

    with pytest.raises(RuntimeError, match="can't be cast"):
        adj -= 2.5


@withCUDA
@withPackage('torch_sparse')
@pytest.mark.parametrize('reduce', ReduceType.__args__)
@pytest.mark.parametrize('transpose', TRANSPOSE)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_torch_sparse_spmm(device, reduce, transpose, is_undirected):
    if is_undirected:
        kwargs = dict(is_undirected=True)
        adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, **kwargs)
    else:
        adj = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], device=device)
    adj = adj.sort_by('col' if transpose else 'row').values

    # Basic:
    x = torch.randn(3, 1, device=device)

    out = _torch_sparse_spmm(adj, x, None, reduce, transpose)
    exp = _scatter_spmm(adj, x, None, reduce, transpose)
    assert out.allclose(exp, atol=1e-6)

    # With non-zero values:
    x = torch.randn(3, 1, device=device)
    value = torch.rand(adj.size(1), device=device)

    out = _torch_sparse_spmm(adj, x, value, reduce, transpose)
    exp = _scatter_spmm(adj, x, value, reduce, transpose)
    assert out.allclose(exp, atol=1e-6)

    # Gradients w.r.t. other:
    x1 = torch.randn(3, 1, device=device, requires_grad=True)
    x2 = x1.detach().requires_grad_()
    grad = torch.randn_like(x1)

    out = _torch_sparse_spmm(adj, x1, None, reduce, transpose)
    out.backward(grad)
    exp = _scatter_spmm(adj, x2, None, reduce, transpose)
    exp.backward(grad)
    assert x1.grad.allclose(x2.grad, atol=1e-6)

    # Gradients w.r.t. value:
    x = torch.randn(3, 1, device=device)
    value1 = torch.rand(adj.size(1), device=device, requires_grad=True)
    value2 = value1.detach().requires_grad_()
    grad = torch.randn_like(x)

    out = _torch_sparse_spmm(adj, x, value1, reduce, transpose)
    out.backward(grad)
    exp = _scatter_spmm(adj, x, value2, reduce, transpose)
    exp.backward(grad)
    assert value1.grad.allclose(value2.grad, atol=1e-6)


@withCUDA
@pytest.mark.parametrize('reduce', ReduceType.__args__)
@pytest.mark.parametrize('transpose', TRANSPOSE)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_torch_spmm(device, reduce, transpose, is_undirected):
    if is_undirected:
        kwargs = dict(is_undirected=True)
        adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, **kwargs)
    else:
        adj = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], device=device)
    adj, perm = adj.sort_by('col' if transpose else 'row')

    # Basic:
    x = torch.randn(3, 2, device=device)

    if ((not x.is_cuda and torch_geometric.typing.WITH_PT20)
            or reduce in ['sum', 'add']):
        out = _TorchSPMM.apply(adj, x, None, reduce, transpose)
        exp = _scatter_spmm(adj, x, None, reduce, transpose)
        assert out.allclose(exp)
    else:
        with pytest.raises(AssertionError):
            _TorchSPMM.apply(adj, x, None, reduce, transpose)

    # With non-zero values:
    x = torch.randn(3, 1, device=device)
    value = torch.rand(adj.size(1), device=device)

    if ((not x.is_cuda and torch_geometric.typing.WITH_PT20)
            or reduce in ['sum', 'add']):
        out = _TorchSPMM.apply(adj, x, value, reduce, transpose)
        exp = _scatter_spmm(adj, x, value, reduce, transpose)
        assert out.allclose(exp)
    else:
        with pytest.raises(AssertionError):
            _TorchSPMM.apply(adj, x, value, reduce, transpose)

    # Gradients w.r.t. other:
    x1 = torch.randn(3, 1, device=device, requires_grad=True)
    x2 = x1.detach().requires_grad_()
    grad = torch.randn_like(x1)

    if reduce in ['sum', 'add']:
        out = _TorchSPMM.apply(adj, x1, None, reduce, transpose)
        out.backward(grad)
        exp = _scatter_spmm(adj, x2, None, reduce, transpose)
        exp.backward(grad)
        assert x1.grad.allclose(x2.grad)
    else:
        with pytest.raises(AssertionError):
            out = _TorchSPMM.apply(adj, x1, None, reduce, transpose)
            out.backward(grad)

    # Gradients w.r.t. value:
    x = torch.randn(3, 1, device=device)
    value1 = torch.rand(adj.size(1), device=device, requires_grad=True)
    grad = torch.randn_like(x)

    with pytest.raises((AssertionError, NotImplementedError)):
        out = _TorchSPMM.apply(adj, x, value1, reduce, transpose)
        out.backward(grad)


@withCUDA
@withoutExtensions
@pytest.mark.parametrize('reduce', ReduceType.__args__)
@pytest.mark.parametrize('transpose', TRANSPOSE)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_spmm(without_extensions, device, reduce, transpose, is_undirected):
    warnings.filterwarnings('ignore', '.*can be accelerated via.*')

    if is_undirected:
        kwargs = dict(is_undirected=True)
        adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, **kwargs)
    else:
        adj = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], device=device)
    adj = adj.sort_by('col' if transpose else 'row').values

    # Basic:
    x = torch.randn(3, 1, device=device)

    with pytest.raises(ValueError, match="to be sorted by"):
        adj.matmul(x, reduce=reduce, transpose=not transpose)

    out = adj.matmul(x, reduce=reduce, transpose=transpose)
    exp = _scatter_spmm(adj, x, None, reduce, transpose)
    assert out.allclose(exp)

    # With non-zero values:
    x = torch.randn(3, 1, device=device)
    value = torch.rand(adj.size(1), device=device)

    with pytest.raises(ValueError, match="'other_value' not supported"):
        adj.matmul(x, reduce=reduce, other_value=value, transpose=transpose)

    out = adj.matmul(x, value, reduce=reduce, transpose=transpose)
    exp = _scatter_spmm(adj, x, value, reduce, transpose)
    assert out.allclose(exp)

    # Gradients w.r.t. other:
    x1 = torch.randn(3, 1, device=device, requires_grad=True)
    x2 = x1.detach().requires_grad_()
    grad = torch.randn_like(x1)

    out = adj.matmul(x1, reduce=reduce, transpose=transpose)
    out.backward(grad)
    exp = _scatter_spmm(adj, x2, None, reduce, transpose)
    exp.backward(grad)
    assert x1.grad.allclose(x2.grad)

    # Gradients w.r.t. value:
    x = torch.randn(3, 1, device=device)
    value1 = torch.rand(adj.size(1), device=device, requires_grad=True)
    value2 = value1.detach().requires_grad_()
    grad = torch.randn_like(x)

    out = adj.matmul(x, value1, reduce=reduce, transpose=transpose)
    out.backward(grad)
    exp = _scatter_spmm(adj, x, value2, reduce, transpose)
    exp.backward(grad)
    assert value1.grad.allclose(value2.grad)


@withCUDA
@pytest.mark.parametrize('reduce', ReduceType.__args__)
@pytest.mark.parametrize('transpose', TRANSPOSE)
@pytest.mark.parametrize('is_undirected', IS_UNDIRECTED)
def test_spspmm(device, reduce, transpose, is_undirected):
    if is_undirected:
        kwargs = dict(device=device, sort_order='row', is_undirected=True)
        adj1 = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    else:
        kwargs = dict(device=device, sort_order='row')
        adj1 = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], **kwargs)

    adj1_dense = adj1.to_dense().t() if transpose else adj1.to_dense()
    adj2 = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order='col',
                     device=device)
    adj2_dense = adj2.to_dense()

    if reduce in ['sum', 'add']:
        out, value = adj1.matmul(adj2, reduce=reduce, transpose=transpose)
        assert isinstance(out, EdgeIndex)
        assert out.is_sorted_by_row
        assert out._sparse_size == (3, 3)
        if not torch_geometric.typing.NO_MKL:
            assert out._indptr is not None
        assert torch.allclose(out.to_dense(value), adj1_dense @ adj2_dense)
    else:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            adj1.matmul(adj2, reduce=reduce, transpose=transpose)


@withCUDA
@withoutExtensions
def test_matmul(without_extensions, device):
    kwargs = dict(sort_order='row', device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    x = torch.randn(3, 1, device=device)
    expected = adj.to_dense() @ x

    out = adj @ x
    assert torch.allclose(out, expected)

    out = adj.matmul(x)
    assert torch.allclose(out, expected)

    out = torch.mm(adj, x)
    assert torch.allclose(out, expected)

    out = torch.matmul(adj, x)
    assert torch.allclose(out, expected)

    if torch_geometric.typing.WITH_PT20:
        out = torch.sparse.mm(adj, x, reduce='sum')
    else:
        with pytest.raises(TypeError, match="got an unexpected keyword"):
            torch.sparse.mm(adj, x, reduce='sum')
        out = torch.sparse.mm(adj, x)
    assert torch.allclose(out, expected)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sparse_row_narrow(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    out = adj.sparse_narrow(dim=0, start=1, length=1)
    assert out.equal(tensor([[0, 0], [0, 2]], device=device))
    assert out.sparse_size() == (1, None)
    assert out.sort_order == 'row'
    assert out._indptr.equal(tensor([0, 2], device=device))

    out = adj.sparse_narrow(dim=0, start=2, length=0)
    assert out.equal(tensor([[], []], device=device))
    assert out.sparse_size() == (0, None)
    assert out.sort_order == 'row'
    assert out._indptr is None

    out = adj.sparse_narrow(dim=1, start=1, length=1)
    assert out.equal(tensor([[0, 2], [0, 0]], device=device))
    assert out.sparse_size() == (3, 1)
    assert out.sort_order == 'row'
    assert out._indptr is None

    out = adj.sparse_narrow(dim=1, start=2, length=0)
    assert out.equal(tensor([[], []], device=device))
    assert out.sparse_size() == (3, 0)
    assert out.sort_order == 'row'
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sparse_col_narrow(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order='col', **kwargs)

    out = adj.sparse_narrow(dim=1, start=1, length=1)
    assert out.equal(tensor([[0, 2], [0, 0]], device=device))
    assert out.sparse_size() == (None, 1)
    assert out.sort_order == 'col'
    assert out._indptr.equal(tensor([0, 2], device=device))

    out = adj.sparse_narrow(dim=1, start=2, length=0)
    assert out.equal(tensor([[], []], device=device))
    assert out.sparse_size() == (None, 0)
    assert out.sort_order == 'col'
    assert out._indptr is None

    out = adj.sparse_narrow(dim=0, start=1, length=1)
    assert out.equal(tensor([[0, 0], [0, 2]], device=device))
    assert out.sparse_size() == (1, 3)
    assert out.sort_order == 'col'
    assert out._indptr is None

    out = adj.sparse_narrow(dim=0, start=2, length=0)
    assert out.equal(tensor([[], []], device=device))
    assert out.sparse_size() == (0, 3)
    assert out.sort_order == 'col'
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sparse_resize(dtype, device):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=dtype, device=device)

    out = adj.sort_by('row')[0].fill_cache_()
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))
    assert out._T_indptr.equal(tensor([0, 1, 3, 4], device=device))
    out = out.sparse_resize_(4, 5)
    assert out.sparse_size() == (4, 5)
    assert out._indptr.equal(tensor([0, 1, 3, 4, 4], device=device))
    assert out._T_indptr.equal(tensor([0, 1, 3, 4, 4, 4], device=device))
    out = out.sparse_resize_(3, 3)
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))
    assert out._T_indptr.equal(tensor([0, 1, 3, 4], device=device))
    out = out.sparse_resize_(None, None)
    assert out.sparse_size() == (None, None)
    assert out._indptr is None
    assert out._T_indptr is None

    out = adj.sort_by('col')[0].fill_cache_()
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))
    assert out._T_indptr.equal(tensor([0, 1, 3, 4], device=device))
    out = out.sparse_resize_(4, 5)
    assert out.sparse_size() == (4, 5)
    assert out._indptr.equal(tensor([0, 1, 3, 4, 4, 4], device=device))
    assert out._T_indptr.equal(tensor([0, 1, 3, 4, 4], device=device))
    out = out.sparse_resize_(3, 3)
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal(tensor([0, 1, 3, 4], device=device))
    assert out._T_indptr.equal(tensor([0, 1, 3, 4], device=device))
    out = out.sparse_resize_(None, None)
    assert out.sparse_size() == (None, None)
    assert out._indptr is None
    assert out._T_indptr is None


def test_to_list():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])
    with pytest.raises(RuntimeError, match="supported for tensor subclasses"):
        adj.tolist()


def test_numpy():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])
    with pytest.raises(RuntimeError, match="supported for tensor subclasses"):
        adj.numpy()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_global_mapping(device, dtype):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, dtype=dtype)
    n_id = tensor([10, 20, 30], device=device, dtype=dtype)

    expected = tensor([[10, 20, 20, 30], [20, 10, 30, 20]], device=device)
    out = n_id[adj]
    assert not isinstance(out, EdgeIndex)
    assert out.equal(expected)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_to_vector(device, dtype):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, dtype=dtype)
    out = adj.to_vector()
    assert not isinstance(out, EdgeIndex)
    assert out.equal(tensor([1, 3, 5, 7], device=device))


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_save_and_load(dtype, device, tmp_path):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    adj.fill_cache_()

    assert adj.sort_order == 'row'
    assert adj._indptr is not None

    path = osp.join(tmp_path, 'edge_index.pt')
    torch.save(adj, path)
    out = fs.torch_load(path)

    assert isinstance(out, EdgeIndex)
    assert out.equal(adj)
    assert out.sort_order == 'row'
    assert out._indptr.equal(adj._indptr)


def _collate_fn(edge_indices: List[EdgeIndex]) -> List[EdgeIndex]:
    return edge_indices


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('pin_memory', [False, True])
def test_data_loader(dtype, num_workers, pin_memory):
    kwargs = dict(dtype=dtype)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)
    adj.fill_cache_()

    loader = torch.utils.data.DataLoader(
        [adj] * 4,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    assert len(loader) == 2
    for batch in loader:
        assert isinstance(batch, list)
        assert len(batch) == 2
        for adj in batch:
            assert isinstance(adj, EdgeIndex)
            assert adj.dtype == dtype
            assert adj.is_shared() != (num_workers == 0) or pin_memory
            assert adj._data.is_shared() != (num_workers == 0) or pin_memory


def test_torch_script():
    class Model(torch.nn.Module):
        def forward(self, x: Tensor, edge_index: EdgeIndex) -> Tensor:
            row, col = edge_index[0], edge_index[1]
            x_j = x[row]
            out = scatter(x_j, col, dim_size=edge_index.num_cols)
            return out

    x = torch.randn(3, 8)
    # Test that `num_cols` gets picked up by making last node isolated.
    edge_index = EdgeIndex([[0, 1, 1, 2], [1, 0, 0, 1]], sparse_size=(3, 3))

    model = Model()
    expected = model(x, edge_index)
    assert expected.size() == (3, 8)

    # `torch.jit.script` does not support inheritance at the `Tensor` level :(
    with pytest.raises(RuntimeError, match="attribute or method 'num_cols'"):
        torch.jit.script(model)

    # A valid workaround is to treat `EdgeIndex` as a regular PyTorch tensor
    # whenever we are in script mode:
    class ScriptableModel(torch.nn.Module):
        def forward(self, x: Tensor, edge_index: EdgeIndex) -> Tensor:
            row, col = edge_index[0], edge_index[1]
            x_j = x[row]
            dim_size: Optional[int] = None
            if (not torch.jit.is_scripting()
                    and isinstance(edge_index, EdgeIndex)):
                dim_size = edge_index.num_cols
            out = scatter(x_j, col, dim_size=dim_size)
            return out

    script_model = torch.jit.script(ScriptableModel())
    out = script_model(x, edge_index)
    assert out.size() == (2, 8)
    assert torch.allclose(out, expected[:2])


@onlyLinux
@withPackage('torch==2.3')
def test_compile_basic():
    import torch._dynamo as dynamo

    class Model(torch.nn.Module):
        def forward(self, x: Tensor, edge_index: EdgeIndex) -> Tensor:
            x_j = x[edge_index[0]]
            out = scatter(x_j, edge_index[1], dim_size=edge_index.num_cols)
            return out

    x = torch.randn(3, 8)
    # Test that `num_cols` gets picked up by making last node isolated.
    edge_index = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 0, 1]],
        sparse_size=(3, 3),
        sort_order='row',
    ).fill_cache_()

    model = Model()
    expected = model(x, edge_index)
    assert expected.size() == (3, 8)

    explanation = dynamo.explain(model)(x, edge_index)
    assert explanation.graph_break_count == 0

    compiled_model = torch.compile(model, fullgraph=True)
    out = compiled_model(x, edge_index)
    assert torch.allclose(out, expected)


@onlyLinux
@withPackage('torch==2.3')
@pytest.mark.skip(reason="Does not work currently")
def test_compile_create_edge_index():
    import torch._dynamo as dynamo

    class Model(torch.nn.Module):
        def forward(self) -> EdgeIndex:
            # Wait for: https://github.com/pytorch/pytorch/issues/117806
            edge_index = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])
            return edge_index

    model = Model()

    explanation = dynamo.explain(model)()
    assert explanation.graph_break_count == 0

    compiled_model = torch.compile(model, fullgraph=True)
    assert compiled_model() is None


if __name__ == '__main__':
    import argparse

    warnings.filterwarnings('ignore', ".*Sparse CSR tensor support.*")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    channels = 128
    num_nodes = 20_000
    num_edges = 200_000

    x = torch.randn(num_nodes, channels, device=args.device)
    edge_index = EdgeIndex(
        torch.randint(0, num_nodes, size=(2, num_edges), device=args.device),
        sparse_size=(num_nodes, num_nodes),
    ).sort_by('row')[0]
    edge_index.fill_cache_()
    adj1 = edge_index.to_sparse_csr()
    adj2 = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
    )

    def edge_index_mm(edge_index, x, reduce):
        return edge_index.matmul(x, reduce=reduce)

    def torch_sparse_mm(adj, x):
        return adj @ x

    def sparse_tensor_mm(adj, x, reduce):
        return adj.matmul(x, reduce=reduce)

    def scatter_mm(edge_index, x, reduce):
        return _scatter_spmm(edge_index, x, reduce=reduce)

    funcs = [edge_index_mm, torch_sparse_mm, sparse_tensor_mm, scatter_mm]
    func_names = ['edge_index', 'torch.sparse', 'SparseTensor', 'scatter']

    for reduce in ['sum', 'mean', 'amin', 'amax']:
        func_args = [(edge_index, x, reduce), (adj1, x), (adj2, x, reduce),
                     (edge_index, x, reduce)]
        print(f"reduce='{reduce}':")

        benchmark(
            funcs=funcs,
            func_names=func_names,
            args=func_args,
            num_steps=100 if args.device == 'cpu' else 1000,
            num_warmups=50 if args.device == 'cpu' else 500,
            backward=args.backward,
        )

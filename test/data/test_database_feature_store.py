"""Unit tests for DatabaseFeatureStore — no real database required."""
import numpy as np
import pytest
import torch

from torch_geometric.data.database_feature_store import LRUFeatureCache
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.testing import FakeDatabaseFeatureStore


def _make_store(rows=None, cache=None):
    """Return a FakeDatabaseFeatureStore seeded with float32 rows.

    rows: dict mapping nid -> 1-D list/array used for group=None, attr='x'.
    """
    data = {}
    for nid, row in (rows or {}).items():
        data[(None, 'x', nid)] = np.array(row, dtype=np.float32)
    return FakeDatabaseFeatureStore(data=data, cache=cache)


def _attr(nids, group=None, attr_name='x'):
    return TensorAttr(group_name=group, attr_name=attr_name,
                      index=torch.tensor(nids, dtype=torch.long))


def test_multi_get_returns_rows_in_input_nid_order():
    store = _make_store({0: [1., 0.], 1: [0., 1.], 2: [2., 2.]})
    attr = _attr([2, 0, 1])
    result = store._multi_get_tensor([attr])
    assert len(result) == 1
    out = result[0]
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], torch.tensor([2., 2.]))
    assert torch.allclose(out[1], torch.tensor([1., 0.]))
    assert torch.allclose(out[2], torch.tensor([0., 1.]))


def test_multi_get_single_nid():
    store = _make_store({7: [3., 4.]})
    attr = _attr([7])
    result = store._multi_get_tensor([attr])
    assert torch.allclose(result[0], torch.tensor([[3., 4.]]))


def test_multi_get_empty_attrs_returns_empty_list():
    store = _make_store({0: [1., 2.]})
    assert store._multi_get_tensor([]) == []


def test_multi_get_no_records_raises_runtime_error():
    store = FakeDatabaseFeatureStore(data={})
    attr = _attr([99])
    with pytest.raises(RuntimeError, match="Could not determine shape"):
        store._multi_get_tensor([attr])


def test_cache_miss_fetches_from_db():
    store = _make_store(
        {
            0: [1., 0.],
            1: [0., 1.]
        },
        cache=LRUFeatureCache(maxsize=100),
    )
    attr = _attr([0, 1])
    store._multi_get_tensor([attr])
    assert store.fetch_call_count == 1


def test_second_call_hits_cache_entirely():
    store = _make_store(
        {
            0: [1., 0.],
            1: [0., 1.]
        },
        cache=LRUFeatureCache(maxsize=100),
    )
    attr = _attr([0, 1])
    store._multi_get_tensor([attr])
    store._multi_get_tensor([attr])
    assert store.fetch_call_count == 1


def test_partial_cache_hit_narrows_fetch_index():
    """Pre-populate cache for nids 0 and 1; request 0,1,2 — only 2 fetched."""
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store(
        {
            0: [1., 0.],
            1: [0., 1.],
            2: [2., 2.]
        },
        cache=cache,
    )
    # warm cache for 0 and 1
    store._multi_get_tensor([_attr([0, 1])])
    assert store.fetch_call_count == 1

    # now request all three — only 2 should be a DB fetch
    store._multi_get_tensor([_attr([0, 1, 2])])
    assert store.fetch_call_count == 2  # one more fetch, for nid=2 only

    # result still correct
    result = store._multi_get_tensor([_attr([0, 1, 2])])[0]
    assert torch.allclose(result[0], torch.tensor([1., 0.]))
    assert torch.allclose(result[2], torch.tensor([2., 2.]))


def test_put_tensor_invalidates_cache_slice():
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store({0: [1., 0.]}, cache=cache)
    attr = _attr([0])
    first = store._multi_get_tensor([attr])[0]  # fills cache
    assert torch.allclose(first, torch.tensor([[1., 0.]]))
    assert store.fetch_call_count == 1

    # overwrite and invalidate cache
    store._put_tensor(torch.tensor([[9., 9.]]), attr)
    second = store._multi_get_tensor([attr])[0]  # must re-fetch from db
    assert store.fetch_call_count == 2
    assert torch.allclose(second, torch.tensor([[9., 9.]]))


def test_remove_tensor_invalidates_cache():
    cache = LRUFeatureCache(maxsize=100)
    store = _make_store({0: [1., 0.], 1: [0., 1.]}, cache=cache)
    store._multi_get_tensor([_attr([0, 1])])  # fill cache
    store._remove_tensor(_attr([0]))  # drop nid=0 from cache
    store._multi_get_tensor([_attr([0])])  # must re-fetch
    assert store.fetch_call_count == 2


def test_multi_attr_two_attrs_same_group():
    data = {
        (None, 'x', 0): np.array([1., 0.], dtype=np.float32),
        (None, 'y', 0): np.array([3], dtype=np.int64),
    }
    store = FakeDatabaseFeatureStore(data=data)
    attrs = [
        TensorAttr(group_name=None, attr_name='x',
                   index=torch.tensor([0], dtype=torch.long)),
        TensorAttr(group_name=None, attr_name='y',
                   index=torch.tensor([0], dtype=torch.long)),
    ]
    results = store._multi_get_tensor(attrs)
    assert len(results) == 2
    assert torch.allclose(results[0], torch.tensor([[1., 0.]]))
    assert results[1][0, 0].item() == 3


def test_get_all_tensor_attrs_deduplicates():
    data = {
        (None, 'x', 0): np.zeros(2, dtype=np.float32),
        (None, 'x', 1): np.zeros(2, dtype=np.float32),
        (None, 'y', 0): np.zeros(1, dtype=np.int64),
    }
    store = FakeDatabaseFeatureStore(data=data)
    attrs = store.get_all_tensor_attrs()
    attr_names = [(a.group_name, a.attr_name) for a in attrs]
    assert (None, 'x') in attr_names
    assert (None, 'y') in attr_names
    assert len(attrs) == 2  # deduplicated

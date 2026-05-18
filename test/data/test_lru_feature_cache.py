"""Unit tests for LRUFeatureCache — no database required."""
import numpy as np
import pytest
import torch

from torch_geometric.data.database_feature_store import LRUFeatureCache
from torch_geometric.data.feature_store import TensorAttr


def _attr(nids, group=None, attr_name='x'):
    return TensorAttr(group_name=group, attr_name=attr_name,
                      index=torch.tensor(nids, dtype=torch.long))


def _row(val: float) -> np.ndarray:
    return np.array([val], dtype=np.float32)


def test_maxsize_zero_raises():
    with pytest.raises(ValueError):
        LRUFeatureCache(maxsize=0)


def test_maxsize_negative_raises():
    with pytest.raises(ValueError):
        LRUFeatureCache(maxsize=-1)


def test_get_returns_only_cached_nids():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(1.), 1: _row(2.)})
    hits = cache.get(_attr([0, 1, 2]))
    assert set(hits.keys()) == {0, 1}
    assert hits[0][0] == pytest.approx(1.)


def test_get_empty_cache_returns_empty_dict():
    cache = LRUFeatureCache(maxsize=10)
    assert cache.get(_attr([0])) == {}


def test_len_and_contains():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(1.), 1: _row(2.)})
    assert len(cache) == 2
    assert (None, 'x', 0) in cache
    assert (None, 'x', 99) not in cache


def test_eviction_removes_lru_entry():
    cache = LRUFeatureCache(maxsize=2)
    cache.put(None, 'x', {0: _row(0.), 1: _row(1.)})
    # access 0 → promotes to MRU; 1 becomes LRU
    cache.get(_attr([0]))
    # insert third entry → 1 should be evicted
    cache.put(None, 'x', {2: _row(2.)})
    assert (None, 'x', 0) in cache
    assert (None, 'x', 2) in cache
    assert (None, 'x', 1) not in cache


def test_re_put_existing_entry_promotes_to_mru():
    cache = LRUFeatureCache(maxsize=2)
    cache.put(None, 'x', {0: _row(0.), 1: _row(1.)})
    cache.put(None, 'x', {0: _row(9.)})  # re-put 0 → promotes it
    cache.put(None, 'x', {2: _row(2.)})  # evicts 1, not 0
    assert (None, 'x', 0) in cache
    assert (None, 'x', 1) not in cache


def test_invalidate_specific_nids():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(0.), 1: _row(1.), 2: _row(2.)})
    cache.invalidate(None, 'x', nids=[0, 2])
    assert (None, 'x', 0) not in cache
    assert (None, 'x', 1) in cache
    assert (None, 'x', 2) not in cache


def test_invalidate_full_slice_wipes_attr():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(0.), 1: _row(1.)})
    cache.put(None, 'y', {0: _row(9.)})
    cache.invalidate(None, 'x', nids=None)
    assert (None, 'x', 0) not in cache
    assert (None, 'x', 1) not in cache
    assert (None, 'y', 0) in cache  # different attr untouched


def test_clear_empties_everything():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(1.), 1: _row(2.)})
    cache.clear()
    assert len(cache) == 0


def test_multi_get_returns_hits_and_missing_attrs():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(1.), 1: _row(2.)})
    attrs = [_attr([0, 1, 2])]
    cached, missing = cache.multi_get(attrs)
    assert (None, 'x') in cached
    assert set(cached[(None, 'x')].keys()) == {0, 1}
    assert len(missing) == 1
    assert missing[0].index.tolist() == [2]  # narrowed to uncached nid


def test_multi_get_fully_cached_attr_not_in_missing():
    cache = LRUFeatureCache(maxsize=10)
    cache.put(None, 'x', {0: _row(1.), 1: _row(2.)})
    _, missing = cache.multi_get([_attr([0, 1])])
    assert missing == []


def test_multi_put_via_multi_get_round_trip():
    cache = LRUFeatureCache(maxsize=10)
    values = {(None, 'x'): {0: _row(5.), 1: _row(6.)}}
    cache.multi_put(values)
    hits = cache.get(_attr([0, 1]))
    assert hits[0][0] == pytest.approx(5.)
    assert hits[1][0] == pytest.approx(6.)

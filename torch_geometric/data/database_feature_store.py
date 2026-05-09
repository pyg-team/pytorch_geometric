import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType, NodeType


def _nids_from_attr(attr: TensorAttr) -> List[int]:
    r"""Coerce ``attr.index`` to a list of Python ints."""
    nids = (attr.index.tolist()
            if isinstance(attr.index, Tensor) else list(attr.index))
    return [int(n) for n in nids]


class FeatureCache(ABC):
    r"""Abstract cache for per-node feature rows (:obj:`np.ndarray`)
    used by :class:`DatabaseFeatureStore`.

    Subclasses implement :meth:`get` (batch read for a single attr) and
    :meth:`put` (batch write for a single ``(group_name, attr_name)`` slice).
    :class:`DatabaseFeatureStore` drives them through :meth:`multi_get` and
    :meth:`multi_put`, which iterate over a list of attrs.

    .. note::
        The ABC makes no size or eviction guarantee — picking an eviction
        policy is the implementation's concern.  Use :class:`LRUFeatureCache`
        for a bounded in-memory default, or write a custom subclass for
        Redis, on-disk, or distributed backends.

    .. note::
        Process safety: in-memory caches are per-process. When a
        :class:`~torch.utils.data.DataLoader` spawns workers, each worker
        receives its own cloned cache state and warms up independently —
        hits in the main process do not benefit workers and vice versa.
        For cross-worker coherence use for example a network-backed
        cache like Redis or share a single store via ``num_workers=0``.
    """
    @abstractmethod
    def get(self, attr: TensorAttr) -> Dict[int, np.ndarray]:
        r"""Return ``{nid: row}`` for the cached node IDs in ``attr.index``,
        where each *row* is a :obj:`np.ndarray`.
        Missing IDs are absent from the dict.
        """

    @abstractmethod
    def put(
        self,
        group_name: Optional[NodeType],
        attr_name: str,
        nid_map: Dict[int, np.ndarray],
    ) -> None:
        r"""Store ``{nid: row}`` for the ``(group_name, attr_name)`` slice."""

    def multi_get(
        self,
        attrs: List[TensorAttr],
    ) -> Tuple[Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]],
               List[TensorAttr]]:
        r"""Look up multiple :class:`TensorAttr` in the cache.

        Returns:
            A ``(cached, missing_attrs)`` pair where *cached* maps
            ``(group_name, attr_name)`` to a ``{nid: row}`` dict of cache
            hits, and *missing_attrs* is the list of attrs still needing a
            database fetch — each with its ``index`` narrowed to the
            uncached node IDs. Fully-cached attrs are omitted from
            *missing_attrs*.
        """
        cached: Dict[Tuple[Optional[NodeType], str], Dict[int,
                                                          np.ndarray]] = {}
        missing: List[TensorAttr] = []
        for attr in attrs:
            hits = self.get(attr)
            if hits:
                cached[(attr.group_name, attr.attr_name)] = hits
            miss_ids = [n for n in _nids_from_attr(attr) if n not in hits]
            if miss_ids:
                narrowed = copy.copy(attr)
                narrowed.index = torch.tensor(miss_ids, dtype=torch.int64)
                missing.append(narrowed)
        return cached, missing

    def multi_put(
        self,
        values: Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]],
    ) -> None:
        r"""Write a ``{(group_name, attr_name): {nid: row}}`` mapping into
        the cache via :meth:`put`. Override this method for efficient
        multi_put.
        """
        for (group_name, attr_name), nid_map in values.items():
            self.put(group_name, attr_name, nid_map)

    @abstractmethod
    def invalidate(
        self,
        group_name: Optional[NodeType],
        attr_name: str,
        nids: Optional[List[int]] = None,
    ) -> None:
        r"""Drop cached rows for the ``(group_name, attr_name)`` slice.

        Args:
            group_name: The node type whose cached rows should be dropped,
                or :obj:`None` for homogeneous stores.
            attr_name: The attribute name whose cached rows should be dropped.
            nids: Specific node IDs to drop, or :obj:`None` to wipe every
                cached row in the slice.
        """

    def clear(self) -> None:
        r"""Wipe every cached row.  Override for native bulk-clear."""
        raise NotImplementedError("Override for native bulk-clear.")


class LRUFeatureCache(FeatureCache):
    r"""Bounded in-memory :class:`FeatureCache` with LRU eviction.

    Backed by a single :class:`collections.OrderedDict` keyed by
    ``(group_name, attr_name, nid)``.  Reads move the entry to the most-
    recently-used end; writes evict the least-recently-used entry once the
    cache exceeds *maxsize*.

    Args:
        maxsize (int): Maximum number of cached rows.  Must be positive.
    """
    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._maxsize = maxsize
        self._data: "OrderedDict[Tuple[Optional[NodeType], str, int], np.ndarray]" = (  # noqa: E501
            OrderedDict())

    def get(self, attr: TensorAttr) -> Dict[int, np.ndarray]:
        hits: Dict[int, np.ndarray] = {}
        for nid in _nids_from_attr(attr):
            key = (attr.group_name, attr.attr_name, nid)
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
                hits[nid] = val
        return hits

    def put(
        self,
        group_name: Optional[NodeType],
        attr_name: str,
        nid_map: Dict[int, np.ndarray],
    ) -> None:
        for nid, value in nid_map.items():
            key = (group_name, attr_name, int(nid))
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def invalidate(
        self,
        group_name: Optional[NodeType],
        attr_name: str,
        nids: Optional[List[int]] = None,
    ) -> None:
        if nids is None:
            doomed = [
                key for key in self._data
                if key[0] == group_name and key[1] == attr_name
            ]
        else:
            doomed = [(group_name, attr_name, int(n)) for n in nids]
        for key in doomed:
            self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Tuple[Optional[NodeType], str, int]) -> bool:
        return key in self._data


class DatabaseFeatureStore(FeatureStore, ABC):
    r"""Abstract base for remote-database-backed feature stores.

    Handles the full mini-batch retrieval loop: cache lookup, database fetch,
    decode, cache fill, and output tensor assembly.  Subclasses implement the
    following hooks:

    * :meth:`_fetch_remote_attrs` — query the database for the node IDs in
      ``attr.index``; return a ``(raw_records, fetched_nids)`` pair.
    * :meth:`_decode_remote_attrs` — convert raw records into an
      :obj:`FeatureTensorType` with one row per ``fetched_nid``.

    Caching is pluggable: pass any :class:`FeatureCache` to the
    constructor (or :obj:`None` to disable).  The base class wires
    :meth:`_cache_get`/:meth:`_cache_put` to the cache's batch API.

    Highly recommended to override :meth:`_multi_fetch_remote_attrs` and
    :meth:`_multi_decode_remote_attrs` to batch multiple attrs into a single
    database round-trip (e.g. one Cypher query for both ``x`` and ``y``).

    Args:
        cache (FeatureCache, optional): Cache implementation. Pass
            :obj:`None` to disable caching. (default: :obj:`None`)
        tensor_attr_cls (TensorAttr, optional): A user-defined
            :class:`TensorAttr` subclass. (default: :obj:`None`)
    """
    def __init__(
        self,
        cache: Optional[FeatureCache] = None,
        tensor_attr_cls: Optional[Any] = None,
    ):
        super().__init__(tensor_attr_cls=tensor_attr_cls)
        self._cache: Optional[FeatureCache] = cache

    def _multi_get_tensor(
        self,
        attrs: List[TensorAttr],
    ) -> List[FeatureTensorType]:
        r"""Fetch all *attrs* in as few database round-trips as possible.

        Supports partial cache hits: for each attr (group_name:str,
        attr_name:str, node_index:Tensor) the cache may serve a subset of the
        requested node IDs, and only the uncached node IDs are included in
        the database round-trip via attrs whose ``index`` has been narrowed
        by :meth:`_cache_get`.

        .. note::
            Override :meth:`_fetch_remote_attrs`, :meth:`_decode_remote_attrs`,
            :meth:`_cache_get`, and :meth:`_cache_put` to customise remote
            access behaviour.  Override :meth:`_multi_fetch_remote_attrs` and
            :meth:`_multi_decode_remote_attrs` to batch multiple attrs into a
            single database round-trip.
        """
        if not attrs:
            return []

        cached: Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]]
        missing_attrs: List[TensorAttr]
        cached, missing_attrs = self._cache_get(attrs)

        # decoded_map: {(group_name, attr_name): (np.ndarray, [fetched_nids])}
        decoded_map: Dict[Tuple[Optional[NodeType], str],
                          Tuple[np.ndarray, List[int]]] = {}
        if missing_attrs:
            fetched = self._multi_fetch_remote_attrs(missing_attrs)
            decoded_map = self._multi_decode_remote_attrs(
                fetched, missing_attrs)
            self._cache_put(
                self._decoded_to_nid_map(decoded_map, missing_attrs))

        result: List[FeatureTensorType] = []
        for attr in attrs:
            key = (attr.group_name, attr.attr_name)
            nids = attr.index.tolist() if isinstance(
                attr.index, Tensor) else list(attr.index)

            nid_to_pos = {int(nid): i for i, nid in enumerate(nids)}

            cached_rows = cached.get(key, {})

            # Determine shape from cache hit or decoded result.
            sample = next(iter(cached_rows.values()), None)
            if (sample is None and key in decoded_map
                    and len(decoded_map[key][0])):
                sample = decoded_map[key][0][0]
            if sample is None:
                raise RuntimeError(
                    f"Could not determine shape for attr '{key}': no "
                    f"cache hits and the database returned no records.")

            shape = ((len(nids), *sample.shape) if sample.ndim > 0 else
                     (len(nids), ))
            out = np.empty(shape, dtype=sample.dtype)

            for nid, row in cached_rows.items():
                out[nid_to_pos[int(nid)]] = row

            if key in decoded_map:
                decoded_arr, attr_nids = decoded_map[key]
                if attr_nids:
                    pos = np.fromiter(
                        (nid_to_pos[int(nid)]
                         for nid in attr_nids if int(nid) in nid_to_pos),
                        dtype=np.int64,
                    )
                    out[pos] = decoded_arr[:len(pos)]

            result.append(torch.from_numpy(out))

        return result

    @abstractmethod
    def _fetch_remote_attrs(
        self,
        attr: TensorAttr,
    ) -> Tuple[Any, List[int]]:
        r"""Fetch a single *attr* from the database. ``attr.index`` may be
        narrowed to uncached node IDs.

        Returns:
            ``(records, fetched_nids)`` — raw DB response for
            :meth:`_decode_remote_attrs`, and the node IDs returned, in the
            order rows will appear in the decoded array.
        """

    @abstractmethod
    def _decode_remote_attrs(
        self,
        records: Any,
        attr: TensorAttr,
    ) -> FeatureTensorType:
        r"""Convert raw *records* from :meth:`_fetch_remote_attrs` into a
        dense array for a single attr.

        Args:
            records: The raw database response returned by
                :meth:`_fetch_remote_attrs`.
            attr (TensorAttr): The attr that was fetched.

        Returns:
            An :obj:`FeatureTensorType` (``np.ndarray`` or
            :class:`torch.Tensor`) where row ``i`` corresponds to the
            ``i``-th ``fetched_nid``
            returned by :meth:`_fetch_remote_attrs`.  :class:`~torch.Tensor`
            returns are coerced to :class:`numpy.ndarray` by
            :meth:`_multi_decode_remote_attrs` before reaching the cache or
            output assembly.
        """

    def _multi_fetch_remote_attrs(
        self,
        attrs: List[TensorAttr],
    ) -> Dict[Tuple[Optional[NodeType], str], Tuple[List[object], List[int]]]:
        """Fetch all attrs individually. Override to batch for fewer
        round-trips when attrs share group_name and index.

        Returns mapping: (group_name, attr_name) -> (records, fetched_nids)
        """
        result: Dict[Tuple[Optional[NodeType], str], Tuple[List[object],
                                                           List[int]]] = {}
        for attr in attrs:
            records, fetched_nids = self._fetch_remote_attrs(attr)
            result[(attr.group_name, attr.attr_name)] = (records, fetched_nids)
        return result

    def _multi_decode_remote_attrs(
        self,
        fetched: Dict[Tuple[Optional[NodeType], str], Tuple[List[object],
                                                            List[int]]],
        attrs: List[TensorAttr],
    ) -> Dict[Tuple[Optional[NodeType], str], Tuple[np.ndarray, List[int]]]:
        """Decode all attrs. Override to optimize batch decoding.

        Coerces :class:`~torch.Tensor` returns from
        :meth:`_decode_remote_attrs` to :class:`numpy.ndarray` so the cache
        and output assembly stay numpy-only.

        Returns mapping:
            (group_name, attr_name) -> (decoded_array, fetched_nids)
        """
        result: Dict[Tuple[Optional[NodeType], str], Tuple[np.ndarray,
                                                           List[int]]] = {}
        for attr in attrs:
            key = (attr.group_name, attr.attr_name)
            records, fetched_nids = fetched[key]
            decoded = self._decode_remote_attrs(records, attr)
            if isinstance(decoded, Tensor):
                decoded = decoded.detach().cpu().numpy()
            result[key] = (decoded, fetched_nids)
        return result

    @staticmethod
    def _decoded_to_nid_map(
        decoded_map: Dict[Tuple[Optional[NodeType], str], Tuple[np.ndarray,
                                                                List[int]]],
        attrs: List[TensorAttr],
    ) -> Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]]:
        r"""Build a ``{(group_name, attr_name): {nid: row}}`` mapping from
        the decoded arrays and their ``fetched_nids``, ready for
        :meth:`_cache_put`.  Attrs missing from *decoded_map* are skipped.
        """
        out: Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]] = {}
        for a in attrs:
            key = (a.group_name, a.attr_name)
            if key not in decoded_map:
                continue
            decoded_arr, fetched_nids = decoded_map[key]
            out[key] = {
                int(nid): decoded_arr[i]
                for i, nid in enumerate(fetched_nids)
            }
        return out

    def _cache_get(
        self,
        attrs: List[TensorAttr],
    ) -> Tuple[Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]],
               List[TensorAttr]]:
        r"""Look up *attrs* in the cache via
        :meth:`FeatureCache.multi_get`.

        Returns:
            ``(cached, missing_attrs)``:

            * ``cached`` — ``{(group_name, attr_name): {nid: row}}`` for
              attrs with at least one hit.
            * ``missing_attrs`` — input attrs (copies) with ``index``
              narrowed to uncached node IDs; fully-cached attrs omitted.

            With no cache: ``({}, list(attrs))``.
        """
        if self._cache is None:
            return {}, list(attrs)
        return self._cache.multi_get(attrs)

    def _cache_put(
        self,
        values: Dict[Tuple[Optional[NodeType], str], Dict[int, np.ndarray]],
    ) -> None:
        r"""Write *values* (``{(group_name, attr_name): {nid: row}}``) into
        the cache via :meth:`FeatureCache.multi_put`.  No-op when
        no cache is configured.
        """
        if self._cache is not None:
            self._cache.multi_put(values)

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""Fetch a single *attr* from the database. ``attr.index`` may be
        narrowed to uncached node IDs.

        Returns:
            The :obj:`FeatureTensorType` for the *attr*, or :obj:`None` if
            the *attr* is not found in the database.
        """
        out = self._multi_get_tensor([attr])
        return out[0] if out else None

    @abstractmethod
    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        ...

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""Write *tensor* via :meth:`_put_tensor_db` and invalidate the
        affected cache slice on success.
        """
        ok = self._put_tensor_db(tensor, attr)
        if ok and self._cache is not None:
            self._cache.invalidate(attr.group_name, attr.attr_name,
                                   _nids_from_attr(attr))
        return ok

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""Delete via :meth:`_remove_tensor_db` and invalidate the affected
        cache slice on success.  ``attr.index is None`` wipes the whole
        slice.
        """
        ok = self._remove_tensor_db(attr)
        if ok and self._cache is not None:
            nids = (_nids_from_attr(attr) if attr.index is not None else None)
            self._cache.invalidate(attr.group_name, attr.attr_name, nids)
        return ok

    @abstractmethod
    def _put_tensor_db(
        self,
        tensor: FeatureTensorType,
        attr: TensorAttr,
    ) -> bool:
        r"""Write *tensor* to the database. Return :obj:`True` on success."""

    @abstractmethod
    def _remove_tensor_db(self, attr: TensorAttr) -> bool:
        r"""Delete the rows identified by *attr* from the database.
        ``attr.index is None`` removes every row in the slice.
        """

    @abstractmethod
    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        ...

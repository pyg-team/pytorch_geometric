r"""This class defines the abstraction for a backend-agnostic feature store.
The goal of the feature store is to abstract away all node and edge feature
memory management so that varying implementations can allow for independent
scale-out.

This particular feature store abstraction makes a few key assumptions:
* The features we care about storing are node and edge features of a graph.
  To this end, the attributes that the feature store supports include a
  `group_name` (e.g. a heterogeneous node name or a heterogeneous edge type),
  an `attr_name` (e.g. `x` or `edge_attr`), and an index.
* A feature can be uniquely identified from any associated attributes specified
  in `TensorAttr`.

It is the job of a feature store implementer class to handle these assumptions
properly. For example, a simple in-memory feature store implementation may
concatenate all metadata values with a feature index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
features in interesting manners based on the provided metadata.

Major TODOs for future implementation:
* Async `put` and `get` functionality
"""
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.typing import FeatureTensorType, NodeType
from torch_geometric.utils.mixin import CastMixin

# We allow indexing with a tensor, numpy array, Python slicing, or a single
# integer index.
IndexType = Union[torch.Tensor, np.ndarray, slice, int]


class _FieldStatus(Enum):
    UNSET = None


@dataclass
class TensorAttr(CastMixin):
    r"""Defines the attributes of a :class:`FeatureStore` tensor.
    It holds all the parameters necessary to uniquely identify a tensor from
    the :class:`FeatureStore`.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. :class:`FeatureStore`
    implementations can define a different ordering by overriding
    :meth:`TensorAttr.__init__`.
    """

    # The group name that the tensor corresponds to. Defaults to UNSET.
    group_name: Optional[NodeType] = _FieldStatus.UNSET

    # The name of the tensor within its group. Defaults to UNSET.
    attr_name: Optional[str] = _FieldStatus.UNSET

    # The node indices the rows of the tensor correspond to. Defaults to UNSET.
    index: Optional[IndexType] = _FieldStatus.UNSET

    # Convenience methods #####################################################

    def is_set(self, key: str) -> bool:
        r"""Whether an attribute is set in :obj:`TensorAttr`."""
        assert key in self.__dataclass_fields__
        return getattr(self, key) != _FieldStatus.UNSET

    def is_fully_specified(self) -> bool:
        r"""Whether the :obj:`TensorAttr` has no unset fields."""
        return all([self.is_set(key) for key in self.__dataclass_fields__])

    def update(self, attr: 'TensorAttr') -> 'TensorAttr':
        r"""Updates an :class:`TensorAttr` with set attributes from another
        :class:`TensorAttr`.
        """
        for key in self.__dataclass_fields__:
            if attr.is_set(key):
                setattr(self, key, getattr(attr, key))
        return self


class AttrView(CastMixin):
    r"""Defines a view of a :class:`FeatureStore` that is obtained from a
    specification of attributes on the feature store. The view stores a
    reference to the backing feature store as well as a :class:`TensorAttr`
    object that represents the view's state.

    Users can create views either using the :class:`AttrView` constructor,
    :meth:`FeatureStore.view`, or by incompletely indexing a feature store.
    For example, the following calls all create views:

    .. code-block:: python

        store[group_name]
        store[group_name].feat
        store[group_name, feat]

    While the following calls all materialize those views and produce tensors
    by either calling the view or fully-specifying the view:

    .. code-block:: python

        store[group_name]()
        store[group_name].feat[index]
        store[group_name, feat][index]
    """
    def __init__(self, store: 'FeatureStore', attr: TensorAttr):
        self.__dict__['_store'] = store
        self.__dict__['_attr'] = attr

    # Advanced indexing #######################################################

    def __getattr__(self, key: Any) -> Union['AttrView', FeatureTensorType]:
        r"""Sets the first unset field of the backing :class:`TensorAttr`
        object to the attribute.

        This allows for :class:`AttrView` to be indexed by different values of
        attributes, in order.
        In particular, for a feature store that we want to index by
        :obj:`group_name` and :obj:`attr_name`, the following code will do so:

        .. code-block:: python

            store[group, attr]
            store[group].attr
            store.group.attr
        """
        out = copy.copy(self)

        # Find the first attribute name that is UNSET:
        attr_name: Optional[str] = None
        for field in out._attr.__dataclass_fields__:
            if getattr(out._attr, field) == _FieldStatus.UNSET:
                attr_name = field
                break

        if attr_name is None:
            raise AttributeError(f"Cannot access attribute '{key}' on view "
                                 f"'{out}' as all attributes have already "
                                 f"been set in this view")

        setattr(out._attr, attr_name, key)

        if out._attr.is_fully_specified():
            return out._store.get_tensor(out._attr)

        return out

    def __getitem__(self, key: Any) -> Union['AttrView', FeatureTensorType]:
        r"""Sets the first unset field of the backing :class:`TensorAttr`
        object to the attribute via indexing.

        This allows for :class:`AttrView` to be indexed by different values of
        attributes, in order.
        In particular, for a feature store that we want to index by
        :obj:`group_name` and :obj:`attr_name`, the following code will do so:

        .. code-block:: python

            store[group, attr]
            store[group][attr]

        """
        return self.__getattr__(key)

    # Setting attributes ######################################################

    def __setattr__(self, key: str, value: Any):
        r"""Supports attribute assignment to the backing :class:`TensorAttr` of
        an :class:`AttrView`.

        This allows for :class:`AttrView` objects to set their backing
        attribute values.
        In particular, the following operation sets the :obj:`index` of an
        :class:`AttrView`:

        .. code-block:: python

            view = store.view(group_name)
            view.index = torch.tensor([1, 2, 3])
        """
        if key not in self._attr.__dataclass_fields__:
            raise ValueError(f"Attempted to set nonexistent attribute '{key}' "
                             f"(acceptable attributes are "
                             f"{self._attr.__dataclass_fields__})")

        setattr(self._attr, key, value)

    def __setitem__(self, key: str, value: Any):
        r"""Supports attribute assignment to the backing :class:`TensorAttr` of
        an :class:`AttrView` via indexing.

        This allows for :class:`AttrView` objects to set their backing
        attribute values.
        In particular, the following operation sets the `index` of an
        :class:`AttrView`:

        .. code-block:: python

            view = store.view(TensorAttr(group_name))
            view['index'] = torch.tensor([1, 2, 3])
        """
        self.__setattr__(key, value)

    # Miscellaneous built-ins #################################################

    def __call__(self) -> FeatureTensorType:
        r"""Supports :class:`AttrView` as a callable to force retrieval from
        the currently specified attributes.

        In particular, this passes the current :class:`TensorAttr` object to a
        GET call, regardless of whether all attributes have been specified.
        It returns the result of this call.
        In particular, the following operation returns a tensor by performing a
        GET operation on the backing feature store:

        .. code-block:: python

            store[group_name, attr_name]()
        """
        attr = copy.copy(self._attr)
        for key in attr.__dataclass_fields__:  # Set all UNSET values to None.
            if not attr.is_set(key):
                setattr(attr, key, None)
        return self._store.get_tensor(attr)

    def __copy__(self) -> 'AttrView':
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_attr'] = copy.copy(out.__dict__['_attr'])
        return out

    def __eq__(self, obj: Any) -> bool:
        r"""Compares two :class:`AttrView` objects by checking equality of
        their :class:`FeatureStore` references and :class:`TensorAttr`
        attributes.
        """
        if not isinstance(obj, AttrView):
            return False
        return self._store == obj._store and self._attr == obj._attr

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(store={self._store}, '
                f'attr={self._attr})')


# TODO (manan, matthias) Ideally, we want to let `FeatureStore` inherit from
# `MutableMapping` to clearly indicate its behavior and usage to the user.
# However, having `MutableMapping` as a base class leads to strange behavior
# in combination with PyTorch and PyTorch Lightning, in particular since these
# libraries use customized logic during mini-batch for `Mapping` base classes.


class FeatureStore(ABC):
    r"""An abstract base class to access features from a remote feature store.

    Args:
        tensor_attr_cls (TensorAttr, optional): A user-defined
            :class:`TensorAttr` class to customize the required attributes and
            their ordering to unique identify tensor values.
            (default: :obj:`None`)
    """
    _tensor_attr_cls: TensorAttr

    def __init__(self, tensor_attr_cls: Optional[Any] = None):
        super().__init__()
        self.__dict__['_tensor_attr_cls'] = tensor_attr_cls or TensorAttr

    # Core (CRUD) #############################################################

    @abstractmethod
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""To be implemented by :class:`FeatureStore` subclasses."""

    def put_tensor(self, tensor: FeatureTensorType, *args, **kwargs) -> bool:
        r"""Synchronously adds a :obj:`tensor` to the :class:`FeatureStore`.
        Returns whether insertion was successful.

        Args:
            tensor (torch.Tensor or np.ndarray): The feature tensor to be
                added.
            *args: Arguments passed to :class:`TensorAttr`.
            **kwargs: Keyword arguments passed to :class:`TensorAttr`.

        Raises:
            ValueError: If the input :class:`TensorAttr` is not fully
                specified.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully-specify the input by "
                             f"specifying all 'UNSET' fields")
        return self._put_tensor(tensor, attr)

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""To be implemented by :class:`FeatureStore` subclasses."""

    def get_tensor(
        self,
        *args,
        convert_type: bool = False,
        **kwargs,
    ) -> FeatureTensorType:
        r"""Synchronously obtains a :class:`tensor` from the
        :class:`FeatureStore`.

        Args:
            *args: Arguments passed to :class:`TensorAttr`.
            convert_type (bool, optional): Whether to convert the type of the
                output tensor to the type of the attribute index.
                (default: :obj:`False`)
            **kwargs: Keyword arguments passed to :class:`TensorAttr`.

        Raises:
            ValueError: If the input :class:`TensorAttr` is not fully
                specified.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully-specify the input by "
                             f"specifying all 'UNSET' fields.")

        tensor = self._get_tensor(attr)
        if convert_type:
            tensor = self._to_type(attr, tensor)
        return tensor

    def _multi_get_tensor(
        self,
        attrs: List[TensorAttr],
    ) -> List[Optional[FeatureTensorType]]:
        r"""To be implemented by :class:`FeatureStore` subclasses."""
        return [self._get_tensor(attr) for attr in attrs]

    def multi_get_tensor(
        self,
        attrs: List[TensorAttr],
        convert_type: bool = False,
    ) -> List[FeatureTensorType]:
        r"""Synchronously obtains a list of tensors from the
        :class:`FeatureStore` for each tensor associated with the attributes in
        :obj:`attrs`.

        .. note::
            The default implementation simply iterates over all calls to
            :meth:`get_tensor`. Implementer classes that can provide
            additional, more performant functionality are recommended to
            to override this method.

        Args:
            attrs (List[TensorAttr]): A list of input :class:`TensorAttr`
                objects that identify the tensors to obtain.
            convert_type (bool, optional): Whether to convert the type of the
                output tensor to the type of the attribute index.
                (default: :obj:`False`)

        Raises:
            ValueError: If any input :class:`TensorAttr` is not fully
                specified.
        """
        attrs = [self._tensor_attr_cls.cast(attr) for attr in attrs]
        bad_attrs = [attr for attr in attrs if not attr.is_fully_specified()]
        if len(bad_attrs) > 0:
            raise ValueError(
                f"The input TensorAttr(s) '{bad_attrs}' are not fully "
                f"specified. Please fully-specify them by specifying all "
                f"'UNSET' fields")

        tensors = self._multi_get_tensor(attrs)
        if convert_type:
            tensors = [
                self._to_type(attr, tensor)
                for attr, tensor in zip(attrs, tensors)
            ]
        return tensors

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""To be implemented by :obj:`FeatureStore` subclasses."""

    def remove_tensor(self, *args, **kwargs) -> bool:
        r"""Removes a tensor from the :class:`FeatureStore`.
        Returns whether deletion was successful.

        Args:
            *args: Arguments passed to :class:`TensorAttr`.
            **kwargs: Keyword arguments passed to :class:`TensorAttr`.

        Raises:
            ValueError: If the input :class:`TensorAttr` is not fully
                specified.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully-specify the input by "
                             f"specifying all 'UNSET' fields.")
        return self._remove_tensor(attr)

    def update_tensor(self, tensor: FeatureTensorType, *args,
                      **kwargs) -> bool:
        r"""Updates a :obj:`tensor` in the :class:`FeatureStore` with a new
        value. Returns whether the update was successful.

        .. note::
            Implementer classes can choose to define more efficient update
            methods; the default performs a removal and insertion.

        Args:
            tensor (torch.Tensor or np.ndarray): The feature tensor to be
                updated.
            *args: Arguments passed to :class:`TensorAttr`.
            **kwargs: Keyword arguments passed to :class:`TensorAttr`.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        self.remove_tensor(attr)
        return self.put_tensor(tensor, attr)

    # Additional methods ######################################################

    @abstractmethod
    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        pass

    def get_tensor_size(self, *args, **kwargs) -> Optional[Tuple[int, ...]]:
        r"""Obtains the size of a tensor given its :class:`TensorAttr`, or
        :obj:`None` if the tensor does not exist.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_set('index'):
            attr.index = None
        return self._get_tensor_size(attr)

    @abstractmethod
    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        r"""Returns all registered tensor attributes."""

    # `AttrView` methods ######################################################

    def view(self, *args, **kwargs) -> AttrView:
        r"""Returns a view of the :class:`FeatureStore` given a not yet
        fully-specified :class:`TensorAttr`.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        return AttrView(self, attr)

    # Helper functions ########################################################

    @staticmethod
    def _to_type(
        attr: TensorAttr,
        tensor: FeatureTensorType,
    ) -> FeatureTensorType:
        if isinstance(attr.index, Tensor) and isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor)
        if isinstance(attr.index, np.ndarray) and isinstance(tensor, Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    # Python built-ins ########################################################

    def __setitem__(self, key: TensorAttr, value: FeatureTensorType):
        r"""Supports :obj:`store[tensor_attr] = tensor`."""
        # CastMixin will handle the case of key being a tuple or TensorAttr
        # object:
        key = self._tensor_attr_cls.cast(key)
        assert key.is_fully_specified()
        self.put_tensor(value, key)

    def __getitem__(self, key: TensorAttr) -> Any:
        r"""Supports pythonic indexing into the :class:`FeatureStore`.

        In particular, the following rules are followed for indexing:

        * A fully-specified :obj:`key` will produce a tensor output.

        * A partially-specified :obj:`key` will produce an :class:`AttrView`
          output, which is a view on the :class:`FeatureStore`. If a view is
          called, it will produce a tensor output from the corresponding
          (partially specified) attributes.
        """
        # CastMixin will handle the case of key being a tuple or TensorAttr:
        attr = self._tensor_attr_cls.cast(key)
        if attr.is_fully_specified():
            return self.get_tensor(attr)
        # If the view is not fully-specified, return a :class:`AttrView`:
        return self.view(attr)

    def __delitem__(self, attr: TensorAttr):
        r"""Supports :obj:`del store[tensor_attr]`."""
        # CastMixin will handle the case of key being a tuple or TensorAttr
        # object:
        attr = self._tensor_attr_cls.cast(attr)
        attr = copy.copy(attr)
        for key in attr.__dataclass_fields__:  # Set all UNSET values to None.
            if not attr.is_set(key):
                setattr(attr, key, None)
        self.remove_tensor(attr)

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self, obj: object) -> bool:
        return id(self) == id(obj)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def _nids_from_attr(attr: TensorAttr) -> List[int]:
    r"""Coerce ``attr.index`` to a list of Python ints."""
    nids = (attr.index.tolist()
            if isinstance(attr.index, Tensor) else list(attr.index))
    return [int(n) for n in nids]


class FeatureCache(ABC):
    r"""Abstract cache for per-node feature rows used by
    :class:`DatabaseFeatureStore`.

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
        r"""Return ``{nid: row}`` for the cached node IDs in ``attr.index``.
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
        the cache via :meth:`put`.
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

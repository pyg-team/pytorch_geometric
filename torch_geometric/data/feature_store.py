r"""
This class defines the abstraction for a backend-agnostic feature store. The
goal of the feature store is to abstract away all node and edge feature memory
management so that varying implementations can allow for independent scale-out.

This particular feature store abstraction makes a few key assumptions:
* The features we care about storing are node and edge features of a graph.
  To this end, the attributes that the feature store supports include a
  `group_name` (e.g. a heterogeneous node name or a heterogeneous edge type),
  an `attr_name` (e.g. `x` or `edge_attr`), and an index.
* A feature can be uniquely identified from any associated attributes specified
  in `TensorAttr`.

It is the job of a feature store implementor class to handle these assumptions
properly. For example, a simple in-memory feature store implementation may
concatenate all metadata values with a feature index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
features in interesting manners based on the provided metadata.

Major TODOs for future implementation:
* Async `put` and `get` functionality
"""
import copy
from abc import abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch

from torch_geometric.typing import FeatureTensorType
from torch_geometric.utils.mixin import CastMixin

_field_status = Enum("FieldStatus", "UNSET")

# We allow indexing with a tensor, numpy array, Python slicing, or a single
# integer index.
IndexType = Union[torch.Tensor, np.ndarray, slice, int]


@dataclass
class TensorAttr(CastMixin):
    r"""Defines the attributes of a class:`FeatureStore` tensor; in particular,
    all the parameters necessary to uniquely identify a tensor from the feature
    store.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. Feature store
    implementor classes can define a different ordering by overriding
    :meth:`TensorAttr.__init__`.
    """

    # The group name that the tensor corresponds to. Defaults to None.
    group_name: Optional[str] = _field_status.UNSET

    # The name of the tensor within its group. Defaults to None.
    attr_name: Optional[str] = _field_status.UNSET

    # The node indices the rows of the tensor correspond to. Defaults to UNSET.
    index: Optional[IndexType] = _field_status.UNSET

    # Convenience methods #####################################################

    def is_set(self, key: str) -> bool:
        r"""Whether an attribute is set in :obj:`TensorAttr`."""
        assert key in self.__dataclass_fields__
        return getattr(self, key) != _field_status.UNSET

    def is_fully_specified(self) -> bool:
        r"""Whether the :obj:`TensorAttr` has no unset fields."""
        return all([self.is_set(key) for key in self.__dataclass_fields__])

    def fully_specify(self):
        r"""Sets all :obj:`UNSET` fields to :obj:`None`."""
        for key in self.__dataclass_fields__:
            if not self.is_set(key):
                setattr(self, key, None)
        return self

    def update(self, attr: 'TensorAttr'):
        r"""Updates an :class:`TensorAttr` with set attributes from another
        :class:`TensorAttr`."""
        for key in self.__dataclass_fields__:
            if attr.is_set(key):
                setattr(self, key, getattr(attr, key))


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
        r"""Sets the first unset field of the backing :class:`TensorAttr` object
        to the attribute. This allows for :class:`AttrView` to be indexed by
        different values of attributes, in order. In particular, for a feature
        store that we want to index by :obj:`group_name` and :obj:`attr_name`,
        the following code will do so:

        .. code-block:: python

            store[group, attr]
            store[group].attr
            store.group.attr
        """
        out = copy.copy(self)

        # Find the first attribute name that is UNSET:
        attr_name: Optional[str] = None
        for field in out._attr.__dataclass_fields__:
            if getattr(out._attr, field) == _field_status.UNSET:
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
        r"""Sets the first unset field of the backing :class:`TensorAttr` object
        to the attribute via indexing. This allows for :class:`AttrView` to be
        indexed by different values of attributes, in order. In particular, for
        a feature store that we want to index by :obj:`group_name` and
        :obj:`attr_name`, the following code will do so:

        .. code-block:: python

            store[group, attr]
            store[group][attr]

        """
        return self.__getattr__(key)

    # Setting attributes ######################################################

    def __setattr__(self, key: str, value: Any):
        r"""Supports attribute assignment to the backing :class:`TensorAttr` of
        an :class:`AttrView`. This allows for :class:`AttrView` objects to set
        their backing attribute values. In particular, the following operation
        sets the :obj:`index` of an :class:`AttrView`:

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
        an :class:`AttrView` via indexing. This allows for :class:`AttrView`
        objects to set their backing attribute values. In particular, the
        following operation sets the `index` of an :class:`AttrView`:

        .. code-block:: python

            view = store.view(TensorAttr(group_name))
            view['index'] = torch.Tensor([1, 2, 3])
        """
        self.__setattr__(key, value)

    # Miscellaneous built-ins #################################################

    def __call__(self) -> FeatureTensorType:
        r"""Supports :class:`AttrView` as a callable to force retrieval from
        the currently specified attributes. In particular, this passes the
        current :class:`TensorAttr` object to a GET call, regardless of whether
        all attributes have been specified. It returns the result of this call.
        In particular, the following operation returns a tensor by performing a
        GET operation on the backing feature store:

        .. code-block:: python

            store[group_name, attr_name]()
        """
        # Set all UNSET values to None:
        out = copy.copy(self)
        out._attr.fully_specify()
        return out._store.get_tensor(out._attr)

    def __copy__(self) -> 'AttrView':
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_attr'] = copy.copy(out.__dict__['_attr'])
        return out

    def __eq__(self, obj: Any) -> bool:
        r"""Compares two :class:`AttrView` objects by checking equality of their
        :class:`FeatureStore` references and :class:`TensorAttr` attributes."""
        if not isinstance(obj, AttrView):
            return False
        return self._store == obj._store and self._attr == obj._attr

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(store={self._store}, '
                f'attr={self._attr})')


class FeatureStore(MutableMapping):
    def __init__(self, attr_cls: Any = TensorAttr):
        r"""Initializes the feature store. Implementor classes can customize
        the ordering and required nature of their :class:`TensorAttr` tensor
        attributes by subclassing :class:`TensorAttr` and passing the subclass
        as :obj:`attr_cls`."""
        super().__init__()
        self._attr_cls = attr_cls

    # Core (CRUD) #############################################################

    @abstractmethod
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""To be implemented by :class:`FeatureStore` subclasses."""
        pass

    def put_tensor(self, tensor: FeatureTensorType, *args, **kwargs) -> bool:
        r"""Synchronously adds a :class:`FeatureTensorType` object to the
        feature store.

        Args:
            tensor (FeatureTensorType): The feature tensor to be added.
            **attr (TensorAttr): Any relevant tensor attributes that correspond
                to the feature tensor. See the :class:`TensorAttr`
                documentation for required and optional attributes. It is the
                job of implementations of a :class:`FeatureStore` to store this
                metadata in a meaningful way that allows for tensor retrieval
                from a :class:`TensorAttr` object.

        Returns:
            bool: Whether insertion was successful.
        """
        attr = self._attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully specify the input by "
                             f"specifying all 'UNSET' fields")
        return self._put_tensor(tensor, attr)

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""To be implemented by :class:`FeatureStore` subclasses."""
        pass

    def get_tensor(self, *args, **kwargs) -> Optional[FeatureTensorType]:
        r"""Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store. Feature store implementors guarantee that the call
        :obj:`get_tensor(put_tensor(tensor, attr), attr) = tensor` holds.

        Args:
            **attr (TensorAttr): Any relevant tensor attributes that correspond
                to the feature tensor. See the :class:`TensorAttr`
                documentation for required and optional attributes. It is the
                job of implementations of a :class:`FeatureStore` to store this
                metadata in a meaningful way that allows for tensor retrieval
                from a :class:`TensorAttr` object.

        Returns:
            FeatureTensorType, optional: a Tensor of the same type as the
            index, or :obj:`None` if no tensor was found.
        """
        def to_type(tensor: FeatureTensorType) -> FeatureTensorType:
            if tensor is None:
                return None
            if (isinstance(attr.index, torch.Tensor)
                    and isinstance(tensor, np.ndarray)):
                return torch.from_numpy(tensor)
            if (isinstance(attr.index, np.ndarray)
                    and isinstance(tensor, torch.Tensor)):
                return tensor.numpy()
            return tensor

        attr = self._attr_cls.cast(*args, **kwargs)
        if isinstance(attr.index, slice):
            if attr.index.start == attr.index.stop == attr.index.step is None:
                attr.index = None

        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully specify the input by "
                             f"specifying all 'UNSET' fields.")

        return to_type(self._get_tensor(attr))

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""To be implemented by :obj:`FeatureStore` subclasses."""
        pass

    def remove_tensor(self, *args, **kwargs) -> bool:
        r"""Removes a :obj:`FeatureTensorType` object from the feature store.

        Args:
            **attr (TensorAttr): Any relevant tensor attributes that correspond
                to the feature tensor. See the :class:`TensorAttr`
                documentation for required and optional attributes. It is the
                job of implementations of a :class:`FeatureStore` to store this
                metadata in a meaningful way that allows for tensor retrieval
                from a :class:`TensorAttr` object.

        Returns:
            bool: Whether deletion was succesful.
        """
        attr = self._attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully specify the input by "
                             f"specifying all 'UNSET' fields.")
        self._remove_tensor(attr)

    def update_tensor(self, tensor: FeatureTensorType, *args,
                      **kwargs) -> bool:
        r"""Updates a :class:`FeatureTensorType` object with a new value.
        implementor classes can choose to define more efficient update methods;
        the default performs a removal and insertion.

        Args:
            tensor (FeatureTensorType): The feature tensor to be updated.
            **attr (TensorAttr): Any relevant tensor attributes that correspond
                to the feature tensor. See the :class:`TensorAttr`
                documentation for required and optional attributes. It is the
                job of implementations of a :class:`FeatureStore` to store this
                metadata in a meaningful way that allows for tensor retrieval
                from a :class:`TensorAttr` object.

        Returns:
            bool: Whether the update was succesful.
        """
        attr = self._attr_cls.cast(*args, **kwargs)
        self.remove_tensor(attr)
        return self.put_tensor(tensor, attr)

    # :obj:`AttrView` methods #################################################

    def view(self, *args, **kwargs) -> AttrView:
        r"""Returns an :class:`AttrView` of the feature store, with the defined
        attributes set."""
        attr = self._attr_cls.cast(*args, **kwargs)
        return AttrView(self, attr)

    # Python built-ins ########################################################

    def __setitem__(self, key: TensorAttr, value: FeatureTensorType):
        r"""Supports store[tensor_attr] = tensor."""
        # CastMixin will handle the case of key being a tuple or TensorAttr
        # object:
        key = self._attr_cls.cast(key)
        # We need to fully specify the key for __setitem__ as it does not make
        # sense to work with a view here:
        key.fully_specify()
        self.put_tensor(value, key)

    def __getitem__(self, key: TensorAttr) -> Any:
        r"""Supports pythonic indexing into the feature store. In particular,
        the following rules are followed for indexing:

        * A fully-specified :obj:`key` will produce a tensor output.

        * A partially-specified :obj:`key` will produce an :class:`AttrView`
          output, which is a view on the :class:`FeatureStore`. If a view is
          called, it will produce a tensor output from the corresponding
          (partially specified) attributes.
        """
        # CastMixin will handle the case of key being a tuple or TensorAttr
        # object:
        attr = self._attr_cls.cast(key)
        if attr.is_fully_specified():
            return self.get_tensor(attr)
        return self.view(attr)

    def __delitem__(self, key: TensorAttr):
        r"""Supports del store[tensor_attr]."""
        # CastMixin will handle the case of key being a tuple or TensorAttr
        # object:
        key = self._attr_cls.cast(key)
        key.fully_specify()
        self.remove_tensor(key)

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self, obj: object) -> bool:
        return id(self) == id(obj)

    @abstractmethod
    def __len__(self):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

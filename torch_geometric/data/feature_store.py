r"""
This class defines the abstraction for a Graph feature store. The goal of a
feature store is to abstract away all node and edge feature memory management
so that varying implementations can allow for independent scale-out.

This particular feature store abstraction makes a few key assumptions:
    * The features we care about storing are all associated with some sort of
        `index`; explicitly for PyG the the index of the node in the graph (or
            the heterogeneous component of the graph it resides in).
    * A feature can uniquely be identified from (a) its index and (b) any other
        associated attributes specified in :obj:`TensorAttr`.

It is the job of a feature store implementor class to handle these assumptions
properly. For example, a simple in-memory feature store implementation may
concatenate all metadata values with a feature index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
features in interesting manners based on the provided metadata.

Major TODOs for future implementation:
* Async `put` and `get` functionality
"""
from abc import abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from torch_geometric.typing import FeatureTensorType
from torch_geometric.utils.mixin import CastMixin


@dataclass
class TensorAttr(CastMixin):
    r"""Defines the attributes of a :obj:`FeatureStore` tensor."""

    # The node indices the rows of the tensor correspond to
    index: Optional[FeatureTensorType] = None

    # The type of the feature tensor (may be used if there are multiple
    # different feature tensors for the same node index)
    tensor_type: Optional[str] = None

    # The type of the nodes that the tensor corresponds to (may be used for
    # hetereogeneous graphs)
    node_type: Optional[str] = None

    # The type of the graph that the nodes correspond to (may be used if a
    # feature store supports multiple graphs)
    graph_type: Optional[str] = None


class AttrView:
    r"""A view of a :obj:`FeatureStore` that is obtained from an incomplete
    specification of attributes. This view stores a reference to the
    originating feature store as well as a :obj:`TensorAttr` object that
    represents the view's (incomplete) state.

    As a result, store[TensorAttr(...)].tensor_type[idx] allows for indexing
    into the store.
    """
    _store: 'FeatureStore'
    attr: TensorAttr

    def __init__(self, store, attr):
        self._store = store
        self.attr = attr

    def __getattr__(self, tensor_type):
        r"""Supports attr_view.attr"""
        self.attr.tensor_type = tensor_type
        return self

    def __getitem__(self, index: FeatureTensorType):
        r"""Supports attr_view.attr[idx]"""
        self.attr.index = index
        return self._store.get_tensor(self.attr)

    def __repr__(self) -> str:
        return f'AttrView(store={self._store}, attr={self.attr})'


class FeatureStore(MutableMapping):
    def __init__(self, backend: Any):
        r"""Initializes the feature store with a specified backend."""
        self.backend = backend

    # Core (CRUD) #############################################################

    @abstractmethod
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""Implemented by :obj:`FeatureStore` subclasses."""
        pass

    def put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""Synchronously adds a :obj:`FeatureTensorType` object to the feature
        store.

        Args:
            tensor (FeatureTensorType): the features to be added.
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the feature tensor. See the :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor retrieval from a
                :obj:`TensorAttr` object.
        Returns:
            bool: whether insertion was successful.
        """
        attr = TensorAttr.cast(attr)
        assert attr.index is not None
        assert attr.index.size(dim=0) == tensor.size(dim=-1)
        return self._put_tensor(tensor, attr)

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""Implemented by :obj:`FeatureStore` subclasses."""
        pass

    def get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""Synchronously obtains a :obj:`FeatureTensorType` object from the
        feature store. Feature store implementors guarantee that the call
        get_tensor(put_tensor(tensor, attr), attr) = tensor.

        Args:
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the tensor to obtain. See :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor retrieval from a
                :obj:`TensorAttr` object.
        Returns:
            FeatureTensorType, optional: a tensor of the same type as the
            index, or None if no tensor was found.
        """
        def maybe_to_torch(x):
            return torch.from_numpy(x) if isinstance(
                attr.index, torch.Tensor) and isinstance(x, np.ndarray) else x

        attr = TensorAttr.cast(attr)
        assert attr.index is not None

        return maybe_to_torch(self._get_tensor(attr))

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""Implemented by :obj:`FeatureStore` subclasses."""
        pass

    def remove_tensor(self, attr: TensorAttr) -> bool:
        r"""Removes a :obj:`FeatureTensorType` object from the feature store.

        Args:
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the tensor to remove. See :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor deletion from a
                :obj:`TensorAttr` object.

        Returns:
            bool: whether deletion was succesful.
        """
        attr = TensorAttr.cast(attr)
        self._remove_tensor(attr)

    def update_tensor(self, tensor: FeatureTensorType,
                      attr: TensorAttr) -> bool:
        r"""Updates a :obj:`FeatureTensorType` object with a new value.
        implementor classes can choose to define more efficient update methods;
        the default performs a removal and insertion.

        Args:
            tensor (FeatureTensorType): the features to be added.
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the old tensor. See :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor update from a
                :obj:`TensorAttr` object.

        Returns:
            bool: whether the update was succesful.
        """
        attr = TensorAttr.cast(attr)
        self.remove_tensor(attr)
        return self.put_tensor(tensor, attr)

    # Python built-ins ########################################################

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(backend={self.backend})'

    def __setitem__(self, key: TensorAttr, value: FeatureTensorType):
        r"""Supports store[tensor_attr] = tensor."""
        key = TensorAttr.cast(key)
        assert key.index is not None
        self.put_tensor(value, key)

    def __getitem__(self, key: TensorAttr):
        r"""Supports store[tensor_attr]. If tensor_attr has index specified,
        will obtain the corresponding features from the store. Otherwise, will
        return an :obj:`AttrView` which can be indexed independently."""
        key = TensorAttr.cast(key)
        if key.index is not None:
            return self.get_tensor(key)
        return AttrView(self, key)

    def __delitem__(self, key: TensorAttr):
        r"""Supports del store[tensor_attr]."""
        key = TensorAttr.cast(key)
        self.remove_tensor(key)

    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        pass

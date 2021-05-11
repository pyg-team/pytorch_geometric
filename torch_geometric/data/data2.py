from typing import Optional, Dict, Any, Tuple, Union
from torch_geometric.typing import NodeType, EdgeType, QueryType

import copy
from itertools import chain
from collections.abc import Sequence, Mapping

import torch
import numpy as np

from .storage import NodeStorage, EdgeStorage, GlobalStorage


class Data(object):
    def __init__(self, dictionary: Optional[Dict[str, Any]] = None, **kwargs):
        self.__dict__['_global_store'] = GlobalStorage()
        self.__dict__['_hetero_store'] = {}

        for key, value in chain(dictionary.items(), kwargs.items()):
            if '__' in key and isinstance(value, dict):
                key = tuple(key.split('__'))

            if isinstance(value, dict):
                self[key].update(value)
            else:
                setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # `data.*` => Link to the `_global_store`.
        return getattr(self._global_store, key)

    def __setattr__(self, key: str, item: Any):
        # `data.* = ...` => Link to the `_global_store`.
        # NOTE: We aim to prevent duplicates in `_hetero_store` keys.
        if key in self._hetero_store:
            raise AttributeError(
                f"'{key}' attribute is already present as a node/edge-type")
        setattr(self._global_store, key, item)

    def __delattr__(self, key: str):
        # `del data.*` => Link to the `_global_store`.
        if key in self._global_store:
            delattr(self._global_store, key)

    def __getitem__(self, *args: Tuple[QueryType]) -> Any:
        # `data[*]` => Link to either `_hetero_store` or `_global_store`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._hetero_store.get(key, None)
        if out is not None:
            return out

        out = self._global_store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, tuple):
            out = EdgeStorage(_key=key, _parent=self)
        else:
            out = NodeStorage(_key=key, _parent=self)

        self._hetero_store[key] = out

        return out

    def __setitem__(self, key: str, item: Any):
        # `data[*] = ...` => Link to `data.* = ...` in `_global_store`.
        setattr(self, key, item)

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[x]` => Link to either `_hetero_store` or `_global_store`.
        key = self._to_canonical(*args)
        if key in self._hetero_store:
            del self._hetero_store[key]
        elif key in self._global_store:
            del self._global_store[key]

    def __copy__(self):
        out = self.__class__()
        out._global_store = copy.copy(self._global_store)
        for key, item in self._hetero_store.items():
            out._hetero_store[key] = copy.copy(item)
            out._hetero_store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        out._global_store = copy.deepcopy(self._global_store, memo)
        for key, item in self._hetero_store.items():
            out._hetero_store[key] = copy.deepcopy(item, memo)
            out._hetero_store._parent = out
        return out

    def _to_canonical(
        self,
        *args: Tuple[QueryType],
    ) -> Union[NodeType, EdgeType]:
        # Converts a given `QueryType` to its "canonical type", i.e.,
        # "incomplete" edge types `(src_node_type, dst_node_type)` will get
        # linked to `(src_node_type, _, dst_node_type)`.
        if len(args) == 1:
            args = args[0]
        if isinstance(args, tuple) and len(args) == 2:
            args = (args[0], '_', args[1])
        return args

    @property
    def is_heterogeneous(self) -> bool:
        return len(self._hetero_store) > 0

    @property
    def is_homogeneous(self) -> bool:
        return not self.is_heterogeneous

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        if self.is_homogeneous:
            info = [size_repr(k, v) for k, v in self._global_store.items()]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
            info2 = [size_repr(k, v, 2) for k, v in self._hetero_store.items()]
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info1 + info2))


def size_repr(key, value, indent=0):
    indent_str = ' ' * indent
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, torch.Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping):
        lines = [indent_str + size_repr(k, v, 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + indent_str + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    return f'{indent_str}{key}={out}'

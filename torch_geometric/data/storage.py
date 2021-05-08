from typing import Any, Optional, Iterable

import copy
import warnings


class Storage(dict):
    # This class extends the Python dictionary class by the following:
    # 1. It allows attribute assignment:
    #    `storage.x = ...` rather than `storage['x'] = ...`
    # 2. We can only iterate over specific keys:
    #    `storage.items('x', 'y')` or `storage.values('x', 'y')
    def __getattr__(self, key: str) -> Any:
        return self.get(key, None)

    def __setattr__(self, key: str, item: Any):
        if item is None and key in self:
            del self[key]
        elif item is not None:
            self[key] = item

    def __delattr__(self, key: str):
        if key in self:
            del self[key]

    def __copy__(self):
        out = self.__class__()
        out.update(self)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__()
        for key, item in self.items():
            out[key] = copy.deepcopy(item, memo)
        return out

    def keys(self, *args) -> Iterable:
        if len(args) > 0:
            return {key: self[key] for key in args if key in self}.keys()
        else:
            return super().keys()

    def values(self, *args) -> Iterable:
        if len(args) > 0:
            return {key: self[key] for key in args if key in self}.values()
        else:
            return super().values()

    def items(self, *args) -> Iterable:
        if len(args) > 0:
            return {key: self[key] for key in args if key in self}.items()
        else:
            return super().items()

    @property
    def num_nodes(self) -> Optional[int]:
        for value in self.values('num_nodes'):
            return value
        for value in self.values('x', 'pos'):
            return value.size(-2)
        for value in self.values('adj'):
            return value.size(0)
        for value in self.values('adj_t'):
            return value.size(1)
        warnings.warn((
            f"Unable to infer 'num_nodes' from attributes {set(self.keys())}. "
            f"Please consider explicitly setting 'num_nodes' as attribute"))
        for value in self.values('edge_index'):
            return int(max(value)) + 1
        for value in self.values('face'):
            return int(max(value)) + 1
        return None

    @property
    def num_edges(self) -> Optional[int]:
        for value in self.values('num_edges'):
            return value
        for value in self.values('edge_index'):
            return value.size(-1)
        for value in self.values('adj', 'adj_t'):
            return value.nnz()
        warnings.warn((
            f"Unable to infer 'num_edges' from attributes {set(self.keys())}. "
            f"Please consider explicitly setting 'num_edges' as attribute"))
        return None


class NodeStorage(Storage):
    @property
    def num_edges(self) -> Optional[int]:
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'num_edges'")


class EdgeStorage(Storage):
    @property
    def num_nodes(self) -> Optional[int]:
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'num_nodes'")

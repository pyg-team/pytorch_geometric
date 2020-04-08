import inspect
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Callable, Set


class Inspector(object):
    def __init__(self, base_class: Any):
        self.base_class: Any = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    def inspect(self, func: Callable,
                pop_first: bool = False) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict({k: v.default for k, v in params.items()})
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names: Optional[List[str]] = None) -> Set[str]:
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys)

    def distribute(self, func: Callable, kwargs: Dict[str, Any]):
        out = {}
        for key, default in self.params[func.__name__].items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = default
            out[key] = data
        return out

    def implements(self, func_name: str) -> bool:
        return func_name in self.base_class.__class__.__dict__.keys()

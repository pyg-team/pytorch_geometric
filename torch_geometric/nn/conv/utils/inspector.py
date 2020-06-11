import re
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
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names: Optional[List[str]] = None) -> Set[str]:
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys)

    def __implements__(self, cls, func_name: str) -> bool:
        if cls.__name__ == 'MessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def implements(self, func_name: str) -> bool:
        return self.__implements__(self.base_class.__class__, func_name)

    def types(self, func_names: Optional[List[str]] = None) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for func in func_names or list(self.params.keys()):
            for key, param in self.params[func].items():
                if param.annotation is inspect.Parameter.empty:
                    v = 'torch.Tensor'
                else:
                    v = str(param)
                    v = re.sub(r'Union\[(.*?), NoneType\]', r'Optional[\1]', v)
                    v = re.split(r': | =', v)[1]
                if key in out and out[key] != v:
                    raise ValueError(
                        (f'Found inconsistent types for argument {key}. '
                         f'Expected type {out[key]} but found type {v}.'))
                out[key] = v
        return out

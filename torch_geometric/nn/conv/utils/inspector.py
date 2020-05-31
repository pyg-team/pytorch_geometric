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

    def to_named_tuple(self, name, func_names: Optional[List[str]] = None):
        used: set = set()
        out: str = f'class {name}(NamedTuple):\n'
        for func in func_names or list(self.params.keys()):
            for key, p in self.params[func].items():
                if key in used:
                    continue
                used.add(key)
                s = re.sub(r'Union\[(.*), NoneType\]', r'Optional[\1]', str(p))
                s = re.sub(r' = .*', r'', s)
                if p.annotation == inspect.Parameter.empty:
                    s = f'{s}: torch.Tensor'
                out += f'    {s}\n'
        return out


def get_type(item):
    if item is None:
        return 'Optional[torch.Tensor]'
    elif isinstance(item, tuple):
        return 'Tuple[' + ', '.join(get_type(v) for v in item) + ']'
    elif isinstance(item, list):
        return 'List[' + get_type(item[0]) + ']'
    else:
        thetype = type(item)
        if thetype.__module__ == 'builtins':
            return thetype.__name__
        else:
            return f'{thetype.__module__}.{thetype.__name__}'

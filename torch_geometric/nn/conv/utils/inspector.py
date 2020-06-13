import re
import inspect
import pyparsing as pp
from itertools import product
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Callable, Set, Tuple


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
        for func_name in func_names or list(self.params.keys()):
            arg_types, _ = parse_types(getattr(self.base_class, func_name))
            for key in self.params[func_name].keys():
                if key in out and out[key] != arg_types[key]:
                    raise ValueError(
                        (f'Found inconsistent types for argument {key}. '
                         f'Expected type {out[key]} but found type '
                         f'{arg_types[key]}.'))
                out[key] = arg_types[key]
        return out


def type_to_tree(src):
    src = src.replace(',', ' ')
    return pp.nestedExpr('[', ']').parseString(f'[{src}]').asList()[0]


def to_optional_(tree):
    for i in range(len(tree)):
        if tree[i] == 'Union' and tree[i + 1][-1] == 'NoneType':
            tree[i], tree[i + 1] = 'Optional', tree[i + 1][:-1]
        if isinstance(tree[i], list):
            tree[i] = to_optional_(tree[i])
    return tree


def render_tree(tree):
    return re.sub(r'\'|\"', '', str(tree)[1:-1]).replace(', [', '[')


def parse_types(func: Callable) -> Tuple[OrderedDict, str]:
    r"""Parses argument and return types of function `func`."""
    source = inspect.getsource(func)
    signature = inspect.signature(func)
    param_names = list(signature.parameters.keys())

    # Parse `# type: (...) -> ...` annotation.
    match = re.search(r'#\s*type:\s*\((.*)\)\s*->\s*(.*)\s*\n', source)
    if match and len(match.groups()) == 2:
        arg_types = []
        arg_type_str, return_type = match.groups()
        i = depth = 0
        for j, char in enumerate(arg_type_str):
            if char == '[':
                depth += 1
            elif char == ']':
                depth += -1
            elif char == ',' and depth == 0:
                arg_types.append(arg_type_str[i:j].strip())
                i = j + 1
        arg_types.append(arg_type_str[i:].strip())
        arg_types = OrderedDict({k: v for k, v in zip(param_names, arg_types)})
        return_type = return_type.split('#')[0].strip()

    else:  # Alternatively, parse annotations using the inspected signature.
        arg_types = OrderedDict()
        for key, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                t = 'torch.Tensor'
            else:
                t = re.split(r': | =', str(param))[1]
                t = render_tree(to_optional_(type_to_tree(t)))
            arg_types[key] = t

        return_type = signature.return_annotation
        if return_type is inspect.Parameter.empty:
            return_type = 'torch.Tensor'
        elif str(return_type)[:6] != '<class':
            return_type = str(return_type).replace('typing.', '')
            return_type = render_tree(to_optional_(type_to_tree(return_type)))
        elif return_type.__module__ == 'builtins':
            return_type = return_type.__name__
        else:
            return_type = f'{return_type.__module__}.{return_type.__name__}'

    # FIXME JIT cannot handle `torch_sparse.tensor.SparseTensor` type.
    return_type = return_type.replace('torch_sparse.tensor.', '')
    for key, arg_type in arg_types.items():
        arg_types[key] = arg_type.replace('torch_sparse.tensor.', '')

    return arg_types, return_type


def resolve_types(types: Dict[str, str]) -> List[str]:
    resolved_types = []
    for t in types.values():
        if t[:5] != 'Union':  # Only consider top-level `Union` for now.
            resolved_types.append([t])
        else:
            instances = []
            tree = type_to_tree(t)[1]
            for i in range(len(tree)):
                if i < len(tree) - 1 and isinstance(tree[i + 1], list):
                    instances.append(render_tree(tree[i:i + 2]))
                elif isinstance(tree[i], str):
                    instances.append(render_tree(tree[i:i + 1]))
            resolved_types.append(instances)

    return list(product(*resolved_types))

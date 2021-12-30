import re
import ast
import inspect
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Callable, Set

from .typing import parse_types


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
            func = getattr(self.base_class, func_name)
            arg_types = parse_types(func)[0][0]
            for key in self.params[func_name].keys():
                if key in out and out[key] != arg_types[key]:
                    raise ValueError(
                        (f'Found inconsistent types for argument {key}. '
                         f'Expected type {out[key]} but found type '
                         f'{arg_types[key]}.'))
                out[key] = arg_types[key]
        return out

    def distribute(self, func_name, kwargs: Dict[str, Any]):
        out = {}
        for key, param in self.params[func_name].items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

    def get_function_ast(self, func_name):
        tree = ast.parse(inspect.getsource(self.base_class))
        for func in tree.body[0].body:
            if func.name == func_name:
                return func
        raise ValueError(f'Looking for func "{func_name}" inside of class '
                         f'"{self.base_class}" but it was not found.')

    def calls_internal(self, func_name, func_called):
        func_ast = self.get_function_ast(func_name)
        for stmt in ast.walk(func_ast):
            if not isinstance(stmt, ast.Call):
                continue
            for call_stmt in ast.walk(stmt):
                if isinstance(call_stmt, ast.Attribute):
                    if isinstance(call_stmt.value, ast.Name):
                        if call_stmt.value.id == 'self':
                            if call_stmt.attr == func_called:
                                return True


def func_header_repr(func: Callable, keep_annotation: bool = True) -> str:
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    if keep_annotation:
        return ''.join(re.split(r'(\).*?:.*?\n)', source,
                                maxsplit=1)[:2]).strip()

    params_repr = ['self']
    for param in signature.parameters.values():
        params_repr.append(param.name)
        if param.default is not inspect.Parameter.empty:
            params_repr[-1] += f'={param.default}'

    return f'def {func.__name__}({", ".join(params_repr)}):'


def func_body_repr(func: Callable, keep_annotation: bool = True) -> str:
    source = inspect.getsource(func)
    body_repr = re.split(r'\).*?:.*?\n', source, maxsplit=1)[1]
    if not keep_annotation:
        body_repr = re.sub(r'\s*# type:.*\n', '', body_repr)
    return body_repr

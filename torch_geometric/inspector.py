import inspect
import re
import sys
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

import torch
from torch import Tensor


class Parameter(NamedTuple):
    name: str
    type: Type
    type_repr: str
    default: Any


class Signature(NamedTuple):
    param_dict: Dict[str, Parameter]
    return_type: Type
    return_type_repr: str


class Inspector:
    r"""Inspects a given class and collects information about its instance
    methods.

    Args:
        cls (Type): The class to inspect.
    """
    def __init__(self, cls: Type):
        self._cls = cls
        self._signature_dict: Dict[str, Signature] = {}
        self._source_dict: Dict[str, str] = {}

    def _get_modules(self, cls: Type) -> List[str]:
        from torch_geometric.nn import MessagePassing

        modules: List[str] = []
        for base_cls in cls.__bases__:
            if base_cls not in {object, torch.nn.Module, MessagePassing}:
                modules.extend(self._get_modules(base_cls))

        modules.append(cls.__module__)
        return modules

    @property
    def _modules(self) -> List[str]:
        return self._get_modules(self._cls)

    @property
    def _globals(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for module in self._modules:
            out.update(sys.modules[module].__dict__)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._cls.__name__})'

    def eval_type(self, value: Any) -> Type:
        r"""Returns the type hint of a string."""
        return eval_type(value, self._globals)

    def type_repr(self, obj: Any) -> str:
        r"""Returns the type hint representation of an object."""
        return type_repr(obj, self._globals)

    def implements(self, func_name: str) -> bool:
        r"""Returns :obj:`True` in case the inspected class implements the
        :obj:`func_name` method.

        Args:
            func_name (str): The function name to check for existence.
        """
        func = getattr(self._cls, func_name, None)
        if not callable(func):
            return False
        return not getattr(func, '__isabstractmethod__', False)

    # Inspecting Method Signatures ############################################

    def inspect_signature(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[Union[str, int]]] = None,
    ) -> Signature:
        r"""Inspects the function signature of :obj:`func` and returns a tuple
        of parameter types and return type.

        Args:
            func (callabel or str): The function.
            exclude (list[int or str]): A list of parameters to exclude, either
                given by their name or index. (default: :obj:`None`)
        """
        if isinstance(func, str):
            func = getattr(self._cls, func)
        assert callable(func)

        if func.__name__ in self._signature_dict:
            return self._signature_dict[func.__name__]

        signature = inspect.signature(func)
        params = [p for p in signature.parameters.values() if p.name != 'self']

        param_dict: Dict[str, Parameter] = {}
        for i, param in enumerate(params):
            if exclude is not None and (i in exclude or param.name in exclude):
                continue

            param_type = param.annotation
            # Mimic TorchScript to auto-infer `Tensor` on non-present types:
            param_type = Tensor if param_type is inspect._empty else param_type

            param_dict[param.name] = Parameter(
                name=param.name,
                type=self.eval_type(param_type),
                type_repr=self.type_repr(param_type),
                default=param.default,
            )

        return_type = signature.return_annotation
        # Mimic TorchScript to auto-infer `Tensor` on non-present types:
        return_type = Tensor if return_type is inspect._empty else return_type

        self._signature_dict[func.__name__] = Signature(
            param_dict=param_dict,
            return_type=self.eval_type(return_type),
            return_type_repr=self.type_repr(return_type),
        )

        return self._signature_dict[func.__name__]

    def get_signature(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> Signature:
        r"""Returns the function signature of the inspected function
        :obj:`func`.

        Args:
            func (callabel or str): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        func_name = func if isinstance(func, str) else func.__name__
        signature = self._signature_dict.get(func_name)
        if signature is None:
            raise IndexError(f"Could not access signature for function "
                             f"'{func_name}'. Did you forget to inspect it?")

        if exclude is None:
            return signature

        param_dict = {
            name: param
            for name, param in signature.param_dict.items()
            if name not in exclude
        }
        return Signature(
            param_dict=param_dict,
            return_type=signature.return_type,
            return_type_repr=signature.return_type_repr,
        )

    def remove_signature(
        self,
        func: Union[Callable, str],
    ) -> Optional[Signature]:
        r"""Removes the inspected function signature :obj:`func`.

        Args:
            func (callabel or str): The function.
        """
        func_name = func if isinstance(func, str) else func.__name__
        return self._signature_dict.pop(func_name, None)

    def get_param_dict(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Parameter]:
        r"""Returns the parameters of the inspected function :obj:`func`.

        Args:
            func (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return self.get_signature(func, exclude).param_dict

    def get_params(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> List[Parameter]:
        r"""Returns the parameters of the inspected function :obj:`func`.

        Args:
            func (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_param_dict(func, exclude).values())

    def get_flat_param_dict(
        self,
        funcs: List[Union[Callable, str]],
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Parameter]:
        r"""Returns the union of parameters of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (list[str or callable]): The functions.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        param_dict: Dict[str, Parameter] = {}
        for func in funcs:
            params = self.get_params(func, exclude)
            for param in params:
                expected = param_dict.get(param.name)
                if expected is not None and param.type != expected.type:
                    raise ValueError(f"Found inconsistent types for argument "
                                     f"'{param.name}'. Expected type "
                                     f"'{expected.type}' but found type "
                                     f"'{param.type}'.")

                if expected is not None and param.default != expected.default:
                    if (param.default is not inspect._empty
                            and expected.default is not inspect._empty):
                        raise ValueError(f"Found inconsistent defaults for "
                                         f"argument '{param.name}'. Expected "
                                         f"'{expected.default}'  but found "
                                         f"'{param.default}'.")

                    default = expected.default
                    if default is inspect._empty:
                        default = param.default

                    param_dict[param.name] = Parameter(
                        name=param.name,
                        type=param.type,
                        type_repr=param.type_repr,
                        default=default,
                    )

                if expected is None:
                    param_dict[param.name] = param

        return param_dict

    def get_flat_params(
        self,
        funcs: List[Union[Callable, str]],
        exclude: Optional[List[str]] = None,
    ) -> List[Parameter]:
        r"""Returns the union of parameters of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (list[str or callable]): The functions.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_flat_param_dict(funcs, exclude).values())

    def get_param_names(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        r"""Returns the parameter names of the inspected function :obj:`func`.

        Args:
            func (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_param_dict(func, exclude).keys())

    def get_flat_param_names(
        self,
        funcs: List[Union[Callable, str]],
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        r"""Returns the union of parameter names of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (list[str or callable]): The functions.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_flat_param_dict(funcs, exclude).keys())

    def collect_param_data(
        self,
        func: Union[Callable, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        r"""Collects the input data of the inspected function :obj:`func`
        according to its function signature from a data blob.

        Args:
            func (callable or str): The function.
            kwargs (dict[str, Any]): The data blob which may serve as inputs.
        """
        out_dict: Dict[str, Any] = {}
        for param in self.get_params(func):
            if param.name not in kwargs:
                if param.default is inspect._empty:
                    raise TypeError(f"Parameter '{param.name}' is required")
                out_dict[param.name] = param.default
            else:
                out_dict[param.name] = kwargs[param.name]
        return out_dict

    # Inspecting Method Bodies ################################################

    def get_source(self, cls: Optional[Type] = None) -> str:
        r"""Returns the source code of :obj:`cls`."""
        from torch_geometric.nn import MessagePassing

        cls = cls or self._cls
        if cls.__name__ in self._source_dict:
            return self._source_dict[cls.__name__]
        if cls in {object, torch.nn.Module, MessagePassing}:
            return ''
        source = inspect.getsource(cls)
        self._source_dict[cls.__name__] = source
        return source

    def get_params_from_method_call(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[Union[int, str]]] = None,
    ) -> Dict[str, Parameter]:
        r"""Parses a method call of :obj:`func` and returns its keyword
        arguments.

        .. note::
            The method is required to be called via keyword arguments in case
            type annotations are not found.

        Args:
            func (callable or str): The function.
            exclude (list[int or str]): A list of parameters to exclude, either
                given by their name or index. (default: :obj:`None`)
        """
        func_name = func if isinstance(func, str) else func.__name__
        param_dict: Dict[str, Parameter] = {}

        # Three ways to specify the parameters of an unknown function header:
        # 1. Defined as class attributes in `{func_name}_type`.
        # 2. Defined via type annotations in `# {func_name}_type: (...)`.
        # 3. Defined via parsing of the function call.

        # (1) Find class attribute:
        if hasattr(self._cls, f'{func_name}_type'):
            type_dict = getattr(self._cls, f'{func_name}_type')
            if not isinstance(type_dict, dict):
                raise ValueError(f"'{func_name}_type' is expected to be a "
                                 f"dictionary (got '{type(type_dict)}')")

            for name, param_type in type_dict.items():
                param_dict[name] = Parameter(
                    name=name,
                    type=self.eval_type(param_type),
                    type_repr=self.type_repr(param_type),
                    default=inspect._empty,
                )
            return param_dict

        # (2) Find type annotation:
        for cls in self._cls.__mro__:
            source = self.get_source(cls)
            match = find_parenthesis_content(source, f'{func_name}_type:')
            if match is not None:
                for arg in split(match, sep=','):
                    name_and_type_repr = re.split(r'\s*:\s*', arg)
                    if len(name_and_type_repr) != 2:
                        raise ValueError(f"Could not parse argument '{arg}' "
                                         f"of '{func_name}_type' annotation")

                    name, type_repr = name_and_type_repr
                    param_dict[name] = Parameter(
                        name=name,
                        type=self.eval_type(type_repr),
                        type_repr=type_repr,
                        default=inspect._empty,
                    )
                return param_dict

        # (3) Parse the function call:
        for cls in self._cls.__mro__:
            source = self.get_source(cls)
            source = remove_comments(source)
            match = find_parenthesis_content(source, f'self.{func_name}')
            if match is not None:
                for i, kwarg in enumerate(split(match, sep=',')):
                    if ('=' not in kwarg and exclude is not None
                            and i in exclude):
                        continue

                    name_and_content = re.split(r'\s*=\s*', kwarg)
                    if len(name_and_content) != 2:
                        raise ValueError(f"Could not parse keyword argument "
                                         f"'{kwarg}' in 'self.{func_name}()'")

                    name, _ = name_and_content

                    if exclude is not None and name in exclude:
                        continue

                    param_dict[name] = Parameter(
                        name=name,
                        type=Tensor,
                        type_repr=self.type_repr(Tensor),
                        default=inspect._empty,
                    )
                return param_dict

        return {}  # (4) No function call found:


def eval_type(value: Any, _globals: Dict[str, Any]) -> Type:
    r"""Returns the type hint of a string."""
    if isinstance(value, str):
        value = typing.ForwardRef(value)
    return typing._eval_type(value, _globals, None)  # type: ignore


def type_repr(obj: Any, _globals: Dict[str, Any]) -> str:
    r"""Returns the type hint representation of an object."""
    def _get_name(name: str, module: str) -> str:
        return name if name in _globals else f'{module}.{name}'

    if isinstance(obj, str):
        return obj

    if obj is type(None):
        return 'None'

    if obj is ...:
        return '...'

    if obj.__module__ == 'typing':  # Special logic for `typing.*` types:
        name = obj._name
        if name is None:  # In some cases, `_name` is not populated.
            name = str(obj.__origin__).split('.')[-1]

        args = getattr(obj, '__args__', None)
        if args is None or len(args) == 0:
            return _get_name(name, obj.__module__)
        if all(isinstance(arg, typing.TypeVar) for arg in args):
            return _get_name(name, obj.__module__)

        # Convert `Union[*, None]` to `Optional[*]`.
        # This is only necessary for old Python versions, e.g. 3.8.
        # TODO Only convert to `Optional` if `Optional` is importable.
        if (name == 'Union' and len(args) == 2
                and any([arg is type(None) for arg in args])):
            name = 'Optional'

        if name == 'Optional':  # Remove `None` from `Optional` arguments:
            args = [arg for arg in obj.__args__ if arg is not type(None)]

        args_repr = ', '.join([type_repr(arg, _globals) for arg in args])
        return f'{_get_name(name, obj.__module__)}[{args_repr}]'

    if obj.__module__ == 'builtins':
        return obj.__qualname__

    return _get_name(obj.__qualname__, obj.__module__)


def find_parenthesis_content(source: str, prefix: str) -> Optional[str]:
    r"""Returns the content of :obj:`{prefix}.*(...)` within :obj:`source`."""
    match = re.search(prefix, source)
    if match is None:
        return None

    offset = source[match.start():].find('(')
    if offset < 0:
        return None

    source = source[match.start() + offset:]

    depth = 0
    for end, char in enumerate(source):
        if char == '(':
            depth += 1
        if char == ')':
            depth -= 1
        if depth == 0:
            content = source[1:end]
            # Properly handle line breaks and multiple white-spaces:
            content = content.replace('\n', ' ')
            content = content.replace('#', ' ')
            content = re.sub(' +', ' ', content)
            content = content.strip()
            return content

    return None


def split(content: str, sep: str) -> List[str]:
    r"""Splits :obj:`content` based on :obj:`sep`.
    :obj:`sep` inside parentheses or square brackets are ignored.
    """
    assert len(sep) == 1
    outs: List[str] = []

    start = depth = 0
    for end, char in enumerate(content):
        if char == '[' or char == '(':
            depth += 1
        elif char == ']' or char == ')':
            depth -= 1
        elif char == sep and depth == 0:
            outs.append(content[start:end].strip())
            start = end + 1
    if start != len(content):  # Respect dangling `sep`:
        outs.append(content[start:].strip())
    return outs


def remove_comments(content: str) -> str:
    content = re.sub(r'\s*#.*', '', content)
    content = re.sub(re.compile(r'r"""(.*?)"""', re.DOTALL), '', content)
    content = re.sub(re.compile(r'"""(.*?)"""', re.DOTALL), '', content)
    content = re.sub(re.compile(r"r'''(.*?)'''", re.DOTALL), '', content)
    content = re.sub(re.compile(r"'''(.*?)'''", re.DOTALL), '', content)
    return content

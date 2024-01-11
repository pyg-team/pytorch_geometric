import inspect
import re
from itertools import chain
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

from torch import Tensor


class Parameter(NamedTuple):
    name: str
    type: Type
    default: Any


class Signature(NamedTuple):
    param_dict: Dict[str, Parameter]
    return_type: Type


class Inspector:
    r"""Inspects a given class and collects information about its instance
    methods.

    Args:
        cls (Type): The class to inspect.
    """
    def __init__(self, cls: Type):
        self._cls = cls
        self._signature_dict: Dict[str, Signature] = {}
        self._body_dict: Dict[str, str] = {}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._cls.__name__})'

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
            func (callabel or str): The function to inspect.
            exclude (list[int or str]): A list of parameters to exclude, either
                given by their name or index. (default: :obj:`None`)
        """
        if isinstance(func, str):
            func = getattr(self._cls, func)

        if func.__name__ in self._signature_dict:
            return self._signature_dict[func.__name__]

        signature = inspect.signature(func)
        params = [p for p in signature.parameters.values() if p.name != 'self']

        param_dict: Dict[str, Parameter] = {}
        for i, param in enumerate(params):
            if exclude is not None and (i in exclude or param.name in exclude):
                continue

            param_type = param.annotation
            param_dict[param.name] = Parameter(
                name=param.name,
                # Mimic TorchScript which auto-infers `Tensor` on empty types:
                type=Tensor if param_type is inspect._empty else param_type,
                default=param.default,
            )

        return_type = signature.return_annotation
        return_type = Tensor if return_type is inspect._empty else return_type

        self._signature_dict[func.__name__] = Signature(
            param_dict, return_type)

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

        param_dict = signature.param_dict
        param_dict = {k: v for k, v in param_dict.items() if k not in exclude}
        return Signature(param_dict, signature.return_type)

    def get_params(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> List[Parameter]:
        r"""Returns the parameters of the inspected function :obj:`func`.

        Args:
            funcs (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        signature = self.get_signature(func, exclude)
        return [param for param in signature.param_dict.values()]

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
        return list(set(chain(self.get_params(f, exclude) for f in funcs)))

    def get_param_names(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        r"""Returns the parameter names of the inspected function :obj:`func`.

        Args:
            funcs (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return [param.name for param in self.get_params(func, exclude)]

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
        return [param.name for param in self.get_flat_params(funcs, exclude)]

    def collect_param_data(
        self,
        func: Union[Callable, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        r"""Collects the input data of the inspected function :obj:`func`
        according to its function signature from a data blob.

        Args:
            func (callabel or str): The function.
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

    def inspect_body(self, func: Union[Callable, str]) -> str:
        r"""Inspects the function body of :obj:`func` and returns its content.

        Args:
            func (callabel or str): The function to inspect.
        """
        if isinstance(func, str):
            func = getattr(self._cls, func)

        if func.__name__ in self._body_dict:
            return self._body_dict[func.__name__]

        self._body_dict[func.__name__] = inspect.getsource(func)
        return self._body_dict[func.__name__]

    def get_body(self, func: Union[Callable, str]) -> str:
        r"""Returns the function body of the inspected function :obj:`func`.

        Args:
            func (callabel or str): The function.
        """
        func_name = func if isinstance(func, str) else func.__name__
        body = self._body_dict.get(func_name)
        if body is None:
            raise IndexError(f"Could not access body for function "
                             f"'{func_name}'. Did you forget to inspect it?")
        return body

    def get_params_from_method_call(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[Union[int, str]] = None,
    ) -> Dict[str, Parameter]:

        if isinstance(func, str):
            func = getattr(self._cls, func)

        # Three ways to specify the parameters of an unknown function header:
        # 1. Defined as class attributes in `{func_name}_type`.
        # 2. Defined via type annotations in `# {func_name}_type: (...)`.
        # 3. Defined via parsing of the function call.

        # (1) Find class attribute:
        if hasattr(self, f'{func.__name__}_type'):
            type_dict = getattr(self, f'{func.__name__}_type')
            if not isinstance(type_dict, dict):
                raise ValueError(f"'{func.__name__}_type' is expected to be a "
                                 f"dictionary (got '{type(type_dict)}')")
            return {
                name: Parameter(name, param_type, inspect._empty)
                for name, param_type in type_dict.items()
            }

        # (2) Find type annotation:
        source = inspect.getsource(func)
        match = find_parenthesis_content(source, f'{func.__name__}_type')
        if match is not None:
            param_dict = Dict[str, Parameter] = {}
            for type_repr in split(match, sep=','):
                name_and_type = re.split(r'\s*:\s*', type_repr)
                if len(name_and_type) != 2:
                    raise ValueError(f"Could not parse the argument "
                                     f"'{type_repr}' of the "
                                     f"'{func.__name__}_type' annotation")
                name, param_type = name_and_type[0], eval(name_and_type[1])
                param_dict[name] = Parameter(name, param_type, inspect._empty)
            return param_dict

        # (3) Parse the function call:
        match = find_parenthesis_content(source, f'self.{func.__name__}')
        if match is not None:
            param_dict = Dict[str, Parameter] = {}
            for i, type_repr in enumerate(split(match, sep=',')):
                if exclude is not None and i in exclude:
                    continue
                name_and_content = re.split(r'\s*=\s*', type_repr)
                if len(name_and_content) != 2:
                    raise ValueError(f"Could not parse the keyword argument "
                                     f"'{type_repr}' of the "
                                     f"'self.{func.__name__}' call")
                name = name_and_content[0]
                if exclude is not None and name in exclude:
                    continue
                param_dict[name] = Parameter(name, Tensor, inspect._empty)
            return param_dict

        return {}  # (4) No function call found:


def find_parenthesis_content(source: str, prefix: str) -> Optional[str]:
    match = re.search(prefix, source)
    if match is None:
        return

    offset = source[match.start():].find('(')
    if offset < 0:
        return

    depth = 0
    for end, char in enumerate(source[match.start() + offset]):
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


def split(content: str, sep: str) -> List[str]:
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
    outs.append(content[start:].strip())
    return outs

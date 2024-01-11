import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Set,
    Tuple,
    Type,
    Union,
)

from torch import Tensor


class Param(NamedTuple):
    type: Type
    default: Any


class Inspector:
    r"""Inspects a given class and collects information about its instance
    methods.

    Args:
        cls (Type): The class to inspect.
    """
    def __init__(self, cls: Type):
        self._cls = cls
        self._types: Dict[str, Tuple[Dict[str, Param], Type]] = {}

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

    def inspect(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[Union[int, str]] = None,
    ) -> Tuple[Dict[str, Type], Type]:
        r"""Inspect the function signature of :obj:`func` and returns a tuple
        of argument types and return type.

        Args:
            func (callabel or str): The function to inspect.
            pop_first (bool, optional): If set to :obj:`True`, will not inspect
                the first argument of the function.
        """
        if isinstance(func, str):
            func = getattr(self._cls, func)

        if func.__name__ in self._types:
            return self._types[func.__name__]

        signature = inspect.signature(func)

        arg_types: Dict[str, Type] = {}
        for arg_name, param in signature.parameters.items():
            if arg_name == 'self':
                continue

            arg_type = param.annotation
            arg_type = Tensor if arg_type is inspect._empty else arg_type
            arg_types[arg_name] = Param(arg_type, param.default)

        if pop_first:
            arg_types.pop(list(arg_types.keys())[0])

        return_type = signature.return_annotation
        return_type = Tensor if return_type is inspect._empty else return_type

        self._types[func.__name__] = (arg_types, return_type)
        return (arg_types, return_type)

    def get_flat_arg_types(
        self,
        funcs: List[Union[Callable, str]],
    ) -> Dict[str, Type]:
        r"""Returns the union of argument types of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (List[str or callable]): The functions to collect information
                from.
        """
        flat_arg_types: Dict[str, Type] = {}
        for func in funcs:
            func_name = func if isinstance(func, str) else func.__name__
            if func_name not in self._types:
                raise ValueError(f"Could not collect arguments for function "
                                 f"'{func_name}'. Did you forget to inspect "
                                 f"it?")
            arg_types = self._types[func_name][0]
            for arg_name, (arg_type, _) in arg_types.items():
                if arg_name not in flat_arg_types:
                    flat_arg_types[arg_name] = arg_type
                elif flat_arg_types[arg_name] != arg_type:
                    raise ValueError(f"Found inconsistent types for argument "
                                     f"'{arg_name}'. Expected type "
                                     f"'{flat_arg_types[arg_name]}' but found "
                                     f"type '{arg_type}'.")
        return flat_arg_types

    def get_flat_arg_names(
        self,
        funcs: List[Union[Callable, str]],
    ) -> Set[str]:
        r"""Returns the union of argument names of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (List[str or callable]): The functions to collect information
                from.
        """
        return set(self.get_flat_arg_types(funcs).keys())

    def collect(
        self,
        func: Union[Callable, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        r"""Collects the inputs of the inspected function :obj:`func` from a
        data blob.

        Args:
            func (callabel or str): The function to collect inputs for.
            kwargs (dict[str, Any]): The data blob to serve inputs.
        """
        func_name = func if isinstance(func, str) else func.__name__
        if func_name not in self._types:
            raise ValueError(f"Could not collect arguments for function "
                             f"'{func_name}'. Did you forget to inspect it?")

        out_dict: Dict[str, Any] = {}
        for arg_name, (arg_type, default) in self._types[func_name][0].items():
            if arg_name not in kwargs:
                if default is inspect._empty:
                    raise TypeError(f"Required parameter '{arg_name}' missing")
                out_dict[arg_name] = default
            else:
                out_dict[arg_name] = kwargs[arg_name]
        return out_dict

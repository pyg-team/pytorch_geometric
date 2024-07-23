import inspect
from abc import ABCMeta
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Set,
    Tuple,
    Union,
    _tp_cache,
    _TypingEmpty,
    get_type_hints,
)

import numpy as np
from jaxtyping._array_types import _MetaAbstractDtype
from typing_extensions import get_args

from torch_geometric.data import Batch, Data
from torch_geometric.data.batch import DynamicInheritance

# from torchtyping.tensor_type import _AnnotatedType
# from torchtyping.typechecker import _check_tensor


class GenericMeta(type):
    pass


def _check(hint: Any, argument_name: str, value: Any) -> None:
    """Checks if the value matches the type hint and raises an error if does not.

    Args:
        hint (Any): The expected type hint.
        argument_name (str): The name of the argument being checked.
        value (Any): The value to be checked.

    Raises:
        TypeError: If the value does not match the type hint.
    """

    def _check_instance(value: Any) -> None:
        if not isinstance(value, Data) and not isinstance(value, Batch):
            raise TypeError(
                f"{value} ({type(value)}) is not a pyg Data or Batch object."
            )

    # Check if all required attributes are present
    def _check_attributes(hint: Any, attributes: Dict[str, Any]) -> None:
        if hint.check_only_specified and set(attributes.keys()) != hint.attributes:
            raise TypeError(
                f"{argument_name} Data attributes  {set(attributes.keys())} do not match \
                    required set {hint.attributes}"
            )

        if not hint.check_only_specified and not hint.attributes.issubset(attributes):
            raise TypeError(
                f"{argument_name} is missing some attributes from {hint.attributes}"
            )

    # If dtype annotations are provided, check them
    def _check_dtypes(hint: Any, value: Any) -> None:
        dtypes = {k: type(v) for k, v in value._store.items()}
        for colname, dt in hint.dtypes.items():
            if isinstance(
                dt, _MetaAbstractDtype
            ):  # Check for a jaxtyping annotation and use its typechecker
                base_cls, *all_metadata = get_args(dt)
                for metadata in all_metadata:
                    if isinstance(metadata, dict) and "__torchtyping__" in metadata:
                        break
                _check_tensor(colname, value._store[colname], base_cls, metadata)
            # Otherwise just check type
            else:
                if not np.issubdtype(dtypes[colname], np.dtype(dt)):
                    raise TypeError(
                        f"{dtypes[colname]} is not a \
                        subtype of {dt} for data/batch \
                        attribute {colname}"
                    )

    _check_instance(value)
    attributes = value._store
    _check_attributes(hint, attributes)
    # If dtype annotations are provided, check them
    if hint.dtype:
        _check_dtypes()


def typecheck(f: Callable, strict: bool = False) -> Callable:
    """Typechecking decorator."""
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Check input arguments
        bound = signature.bind(*args, **kwargs)
        for argument_name, value in bound.arguments.items():
            hint = hints[argument_name]
            if argument_name in hints and (isinstance(hint, (DataMeta, BatchMeta))):
                _check(hint, argument_name, value)
            elif (
                argument_name in hints
                and not (isinstance(hint, (DataMeta, BatchMeta)))
                and strict
            ):
                if not isinstance(value, hint):
                    raise TypeError(f"{argument_name} is not {hint} {type(value)}")
        # Check return values
        if "return" in hints.keys():
            out_hint = hints["return"]
            if isinstance(out_hint, (DataMeta, BatchMeta)):
                out = f(*args, **kwargs)
                _check(out_hint, "return", out)
                return out
        return f(*args, **kwargs)

    return wrapper


def _resolve_type(t: Any) -> Any:
    """Resolves the underlying type of a given type hint, supporting various
    type hinting constructs such as NewType, AnnotatedType, and generic types.

    Adapted from dataenforce contribution by @martijnentink:
    https://github.com/CedricFR/dataenforce/pull/5


    Args:
        t (Any): The type hint to resolve.

    Returns:
        Any: The resolved underlying type.

    Examples:
        >>> from typing import List
        >>> _resolve_type(List[int])
        <class 'list'>

        >>> from typing import NewType
        >>> UserId = NewType('UserId', int)
        >>> _resolve_type(UserId)
        <class 'int'>
    """
    if isinstance(t, _MetaAbstractDtype):  # support for NewType in type hinting
        return t
    if hasattr(t, "__supertype__"):
        return _resolve_type(t.__supertype__)
    if hasattr(t, "__origin__"):  # support for typing.List and typing.Dict
        return _resolve_type(t.__origin__)
    return t


class DataMeta(GenericMeta, ABCMeta):
    """Metaclass for the `DataT` type, combining generic type support and dynamic inheritance.

    This metaclass is used to define the `DataT` type, which serves as an annotation for
    `torch_geometric.data.Data`. It combines the functionality of `GenericMeta` for generic
    type support and `DynamicInheritance` for dynamic inheritance capabilities.

    Inherits from:
        GenericMeta: Provides support for generic types.
        DynamicInheritance: Allows dynamic inheritance of attributes and methods.

    Attributes:
        None specific to DataMeta.

    Methods:
        None specific to DataMeta.

    Example:
        >>> class MyBatch(BatchT):
        ...     pass
    """

    def __new__(metacls, name, bases, namespace, **kargs):
        return super().__new__(metacls, name, bases, namespace)

    @_tp_cache
    def __getitem__(self, parameters):
        if hasattr(self, "__origin__") and (
            self.__origin__ is not None or self._gorg is not DataT
        ):
            return super().__getitem__(parameters)
        if parameters == ():
            return super().__getitem__((_TypingEmpty,))
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        parameters = list(parameters)

        check_only_specified = True
        if parameters[-1] is ...:
            check_only_specified = False
            parameters.pop()

        attributes, dtypes = _get_attribute_dtypes(parameters)

        meta = DataMeta(self.__name__, self.__bases__, {})
        meta.check_only_specified = check_only_specified
        meta.attributes = attributes
        meta.dtypes = dtypes
        return meta


class BatchMeta(GenericMeta, DynamicInheritance):
    """Metaclass for the `BatchT` type, combining generic type support and dynamic inheritance.

    This metaclass is used to define the `BatchT` type, which serves as an annotation for
    `torch_geometric.data.Batch`. It combines the functionality of `GenericMeta` for generic
    type support and `DynamicInheritance` for dynamic inheritance capabilities.

    Inherits from:
        GenericMeta: Provides support for generic types.
        DynamicInheritance: Allows dynamic inheritance of attributes and methods.

    Attributes:
        None specific to BatchMeta.

    Methods:
        None specific to BatchMeta.

    Example:
        >>> class MyBatch(BatchT):
        ...     pass
    """

    def __new__(metacls, name, bases, namespace, **kargs):
        return super().__new__(metacls, name, bases, namespace)

    @_tp_cache
    def __getitem__(self, parameters):
        if hasattr(self, "__origin__") and (
            self.__origin__ is not None or self._gorg is not BatchT
        ):
            return super().__getitem__(parameters)
        if parameters == ():
            return super().__getitem__((_TypingEmpty,))
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        parameters = list(parameters)

        check_only_specified = True
        if parameters[-1] is ...:
            check_only_specified = False
            parameters.pop()

        attributes, dtypes = _get_attribute_dtypes(parameters)

        meta = BatchMeta(self.__name__, self.__bases__, {})
        meta.check_only_specified = check_only_specified
        meta.attributes = attributes
        meta.dtypes = dtypes
        return meta


def _get_attribute_dtypes(
    p: Union[str, slice, list, set, DataMeta, BatchMeta]
) -> Tuple[Set[str], Dict[str, Any]]:
    """Recursively extracts attribute names and their corresponding data types from the input.

    Args:
        p (Union[str, slice, list, set, DataMeta]): The input parameter which can be a string,
            slice, list, set, or an instance of DataMeta.

    Returns:
        Tuple[Set[str], Dict[str, Any]]: A tuple containing a set of attribute names and a
            dictionary mapping attribute names to their data types.

    Raises:
        TypeError: If the input parameter `p` is not of the expected types.
    """
    attributes = set()
    dtypes = {}
    if isinstance(p, str):
        attributes.add(p)
    elif isinstance(p, slice):
        attributes.add(p.start)
        stop_type = _resolve_type(p.stop)
        dtypes[p.start] = stop_type
    elif isinstance(p, (list, set)):
        for el in p:
            subattributes, subdtypes = _get_attribute_dtypes(el)
            attributes |= subattributes
            dtypes.update(subdtypes)
    elif isinstance(p, DataMeta) or isinstance(p, BatchMeta):
        subattributes, subdtypes = _get_attribute_dtypes(p)
        attributes |= subattributes
        dtypes.update(subdtypes)
    else:
        raise TypeError(
            "DataT[attr1, attr2, ...]: each attribute must be \
            a string, list or set."
        )
    return attributes, dtypes


class DataT(Data, extra=Generic, metaclass=DataMeta):
    """Defines type DataT to serve as annotation for `torch_geometric.data.Data`."""

    __slots__ = ()
    check_only_specified = None
    attributes = set()
    dtypes = {}

    def __new__(cls, *args, **kwds):
        if not hasattr(cls, "_gorg") or cls._gorg is DataT:
            raise TypeError(
                "Type 'GraphT' cannot be instantiated directly. "
                "It is intended to be used as a type annotation only."
            )


class BatchT(Batch, extra=Generic, metaclass=BatchMeta):
    """Defines type BatchT to serve as annotation for `torch_geometric.data.Batch`."""

    __slots__ = ()
    check_only_specified = None
    attributes = set()
    dtypes = {}

    def __new__(cls, *args, **kwds):
        if not hasattr(cls, "_gorg") or cls._gorg is BatchT:
            raise TypeError(
                "Type 'BatchT' cannot be instantiated directly. "
                "It is intended to be used as a type annotation only."
            )

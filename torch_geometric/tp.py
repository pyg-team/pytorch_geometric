import inspect
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
from torchtyping.tensor_type import _AnnotatedType
from torchtyping.typechecker import _check_tensor
from typing_extensions import get_args

from torch_geometric.data import Data

try:
    from typing import GenericMeta  # Python 3.6
except ImportError:  # Python 3.7

    class GenericMeta(type):
        pass


def typecheck(f: Callable) -> Callable:
    """Typechecking decorator."""
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for argument_name, value in bound.arguments.items():
            hint = hints[argument_name]
            if argument_name in hints and isinstance(hint, DataMeta):
                if not isinstance(value, Data):
                    raise TypeError(f"{value} is not a pyg Data object")
                attributes = value._store
                if (hint.check_only_specified
                        and set(attributes.keys()) != hint.attributes):
                    raise TypeError(f"{argument_name} Data attributes \
                        {set(attributes.keys())} do not match \
                         required set {hint.attributes}")

                if (not hint.check_only_specified
                        and not hint.attributes.issubset(attributes)):
                    raise TypeError(f"{argument_name} is missing some \
                        attributes from {hint.attributes}")
                # If dtype annotations are provided, check them
                if hint.dtypes:
                    dtypes = {k: type(v) for k, v in value._store.items()}
                    for colname, dt in hint.dtypes.items():
                        # Check for a torchtyping annotation and use its
                        # typechecker
                        if isinstance(dt, _AnnotatedType):
                            base_cls, *all_metadata = get_args(dt)
                            for metadata in all_metadata:
                                if isinstance(
                                        metadata, dict
                                ) and "__torchtyping__" in metadata:
                                    break
                            _check_tensor(colname, value._store[colname],
                                          base_cls, metadata)
                        # Otherwise just check type
                        else:
                            if not np.issubdtype(dtypes[colname],
                                                 np.dtype(dt)):
                                raise TypeError(f"{dtypes[colname]} is not a \
                                    subtype of {dt} for graph \
                                    attribute {colname}")

        return f(*args, **kwargs)

    return wrapper


def _resolve_type(t: Any) -> Any:
    """Adapted from dataenforce contib by @martijnentink:
    https://github.com/CedricFR/dataenforce/pull/5"""
    # support for NewType in type hinting
    if isinstance(t, _AnnotatedType):
        return t
    if hasattr(t, "__supertype__"):
        return _resolve_type(t.__supertype__)
    # support for typing.List and typing.Dict
    if hasattr(t, "__origin__"):
        return _resolve_type(t.__origin__)
    return t


class DataMeta(GenericMeta):
    """Metaclass for Data (internal)."""
    def __new__(metacls, name, bases, namespace, **kargs):
        return super().__new__(metacls, name, bases, namespace)

    @_tp_cache
    def __getitem__(self, parameters):
        if hasattr(self, "__origin__") and (self.__origin__ is not None
                                            or self._gorg is not Graph):
            return super().__getitem__(parameters)
        if parameters == ():
            return super().__getitem__((_TypingEmpty, ))
        if not isinstance(parameters, tuple):
            parameters = (parameters, )
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


def _get_attribute_dtypes(
    p: Union[str, slice, list, set,
             DataMeta]) -> Tuple[Set[str], Dict[str, Any]]:
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
    elif isinstance(p, DataMeta):
        attributes |= _get_attribute_dtypes(p)[0]
        dtypes.update(_get_attribute_dtypes(p)[1])
    else:
        raise TypeError(
            "Dataset[col1, col2, ...]: each col must be a string, list or set."
        )
    return attributes, dtypes


class Graph(type(Data), extra=Generic, metaclass=DataMeta):
    """Defines type Graph to serve as annotation for PyG Data."""
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if not hasattr(cls, "_gorg") or cls._gorg is Graph:
            raise TypeError("Type Graph cannot be instantiated.")
        return _generic_new(Data, cls, *args, **kwds)


class GraphBatch(type(Data), extra=Generic, metaclass=DataMeta):
    """Defines type Graph to serve as annotation for PyG Data."""
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if not hasattr(cls, "_gorg") or cls._gorg is Graph:
            raise TypeError("Type Graph cannot be instantiated.")
        return _generic_new(GraphBatch, cls, *args, **kwds)

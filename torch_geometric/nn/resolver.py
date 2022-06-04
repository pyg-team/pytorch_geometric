import inspect
from typing import Any, List, Union

import torch
from torch import Tensor


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], query: Union[Any, str], *args, **kwargs):
    if query is None or not isinstance(query, str):
        return query

    query = normalize_string(query)
    for cls in classes:
        if query == normalize_string(cls.__name__):
            if inspect.isclass(cls):
                return cls(*args, **kwargs)
            else:
                return cls

    return ValueError(
        f"Could not resolve '{query}' among the choices "
        f"{set(normalize_string(cls.__name__) for cls in classes)}")


# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    ]
    acts += [
        swish,
    ]
    return resolver(acts, query, *args, **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str] = 'mean', *args, **kwargs):
    from . import aggr

    # NOTE: Base class name: Aggregation, subclasses name: *Aggregation
    base_cls = aggr.Aggregation
    if isinstance(query, str):
        if normalize_string(query) == 'add':
            query = 'sum'
        base_cls_name = normalize_string(base_cls.__name__)
        if base_cls_name not in normalize_string(query):
            query += base_cls_name
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    return resolver(aggrs, query, *args, **kwargs)

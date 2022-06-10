import inspect
from typing import Any, List, Optional, Union

from torch import Tensor


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], query: Union[Any, str],
             base_cls: Optional[Any], *args, **kwargs):

    if query is None or not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    base_cls_repr = normalize_string(base_cls.__name__) if base_cls else ''
    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                return cls(*args, **kwargs)
            else:
                return cls

    return ValueError(f"Could not resolve '{query}' among the choices "
                      f"{set(cls.__name__ for cls in classes)}")


# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    import torch
    base_cls = torch.nn.Module

    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    return resolver(acts, query, base_cls, *args, **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str], *args, **kwargs):
    import torch_geometric.nn.aggr as aggrs
    base_cls = aggrs.Aggregation

    aggrs = [
        aggr for aggr in vars(aggrs).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    return resolver(aggrs, query, base_cls, *args, **kwargs)

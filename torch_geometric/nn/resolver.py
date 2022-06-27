import inspect
from typing import Any, Dict, List, Optional, Union

from torch import Tensor


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], class_dict: Dict[str, Any],
             query: Union[Any, str], base_cls: Optional[Any], *args, **kwargs):

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    base_cls_repr = normalize_string(base_cls.__name__) if base_cls else ''

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


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
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, *args, **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str], *args, **kwargs):
    import torch_geometric.nn.aggr as aggr
    base_cls = aggr.Aggregation
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    aggr_dict = {
        'add': aggr.SumAggregation,
    }
    return resolver(aggrs, aggr_dict, query, base_cls, *args, **kwargs)

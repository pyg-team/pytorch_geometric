import inspect
from typing import Any, Dict, List, Optional, Union

from torch import Tensor


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def camel_to_snake(s: str) -> str:
    import re
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


# Resolvers ###################################################################


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

    choices = set(cls.__name__ for cls in classes + list(class_dict.values()))
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


# Reverse Resolvers ###########################################################


def reverse_resolver(classes: List[Any], class_dict: Dict[Any, str],
                     query: Union[Any, str], base_cls: Optional[Any]):

    if isinstance(query, str):
        return query

    assert callable(query)
    query_cls_repr = camel_to_snake(query.__class__.__name__)

    if not base_cls:
        return query_cls_repr

    base_cls_repr = camel_to_snake(base_cls.__name__)

    if not isinstance(query, base_cls):
        choices = {base_cls_repr} | set(
            camel_to_snake(cls.__name__).replace(base_cls_repr, '').strip('_')
            for cls in classes) | set(class_dict.values())
        choices.remove('')
        raise ValueError(
            f"Could not resolve '{query}' among choices {choices}")

    for cls, repr in class_dict.items():
        if isinstance(query, cls):
            return repr

    repr = query_cls_repr.replace(base_cls_repr, '').strip("_")
    return repr if repr else base_cls_repr


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


# Normalization Resolver ######################################################


def normalization_resolver(query: Union[Any, str], *args, **kwargs):
    import torch

    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, *args, **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str], *args, reverse: bool = False,
                         **kwargs):
    import torch_geometric.nn.aggr as aggr
    base_cls = aggr.Aggregation
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    if not reverse:
        aggr_dict = {
            'add': aggr.SumAggregation,
        }
        return resolver(aggrs, aggr_dict, query, base_cls, *args, **kwargs)
    else:
        aggr_dict = {
            aggr.Set2Set: 'set2set',
        }
        return reverse_resolver(aggrs, aggr_dict, query, base_cls)

import inspect
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from . import basic, lstm
from .base import Aggregation


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


def get_vars_dict(objs):
    d = {}
    for obj in objs:
        d.update(vars(obj))
    return d


def aggregation_resolver(query: Union[Any, str] = 'mean', *args, **kwargs):
    # NOTE: Base class name: Aggregation, subclasses name: *Aggregation
    base_cls = Aggregation
    if isinstance(query, str):
        if normalize_string(query) == 'add':
            query = 'sum'
        base_cls_name = normalize_string(base_cls.__name__)
        if base_cls_name not in normalize_string(query):
            query += base_cls_name
    aggrs = [
        aggr for aggr in get_vars_dict([basic, lstm]).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    return resolver(aggrs, query, *args, **kwargs)


class MultiAggregation(Aggregation):
    def __init__(self, aggrs: List[Union[Aggregation, str]]):
        super().__init__()
        assert type(aggrs) == list, 'aggrs should be a list'
        assert len(aggrs) > 0, f'{len(aggrs)} aggr is passed'
        self.aggrs = list(map(aggregation_resolver, aggrs))

    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        outs = []
        for aggr in self.aggrs:
            tmp = aggr(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)
            outs.append(tmp)
        out = self.combine(outs)
        return out

    def combine(self, inputs: List[Tensor]) -> Tensor:
        r"""Combines the outputs from multiple aggregations into a single
        representation."""
        assert len(inputs) > 0
        return torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self.aggrs)})'

"""A module containing aggregation classes.

Example usage::

>>> import torch
>>> from torch_geometric.nn.aggr import aggregation_resolver
>>> aggregation = aggregation_resolver.make("mean")
>>> x = torch.randn(6, 16)
>>> index = torch.tensor([0, 0, 1, 1, 1, 2])
>>> aggregation(x, index)
"""
from class_resolver import ClassResolver

from .base import Aggregation
from .basic import (
    MeanAggregation,
    SumAggregation,
    MaxAggregation,
    MinAggregation,
    VarAggregation,
    StdAggregation,
    SoftmaxAggregation,
    PowerMeanAggregation,
)

__all__ = classes = [
    # Abstract Class
    'Aggregation',
    # Utilities
    'aggregation_resolver',
    # Concrete Classes
    'MeanAggregation',
    'SumAggregation',
    'MaxAggregation',
    'MinAggregation',
    'VarAggregation',
    'StdAggregation',
    'SoftmaxAggregation',
    'PowerMeanAggregation',
]

aggregation_resolver = ClassResolver.from_subclasses(Aggregation)

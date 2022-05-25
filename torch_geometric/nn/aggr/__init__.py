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

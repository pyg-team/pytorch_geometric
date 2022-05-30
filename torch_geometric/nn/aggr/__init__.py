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
from .lstm import LSTMAggregation

__all__ = classes = [
    'Aggregation',
    'MeanAggregation',
    'SumAggregation',
    'MaxAggregation',
    'MinAggregation',
    'VarAggregation',
    'StdAggregation',
    'SoftmaxAggregation',
    'PowerMeanAggregation',
    'LSTMAggregation',
]

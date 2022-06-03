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
from .set2set import Set2Set

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
    'Set2Set',
]

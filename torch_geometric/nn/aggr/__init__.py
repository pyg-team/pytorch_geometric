from .base import Aggregation
from .multi import MultiAggregation
from .basic import (
    MeanAggregation,
    SumAggregation,
    AddAggregation,
    MaxAggregation,
    MinAggregation,
    MulAggregation,
    VarAggregation,
    StdAggregation,
    SoftmaxAggregation,
    PowerMeanAggregation,
)
from .lstm import LSTMAggregation
from .set2set import Set2Set

__all__ = classes = [
    'Aggregation',
    'MultiAggregation',
    'MeanAggregation',
    'SumAggregation',
    'AddAggregation',
    'MaxAggregation',
    'MinAggregation',
    'MulAggregation',
    'VarAggregation',
    'StdAggregation',
    'SoftmaxAggregation',
    'PowerMeanAggregation',
    'LSTMAggregation',
    'Set2Set',
]

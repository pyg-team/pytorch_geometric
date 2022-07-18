from .base import Aggregation
from .multi import MultiAggregation
from .basic import (
    MeanAggregation,
    SumAggregation,
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
from .scaler import DegreeScalerAggregation
from .sort import SortAggr
from .gmt import GraphMultisetTransformer
from .attention import AttentionalAggregation

__all__ = classes = [
    'Aggregation',
    'MultiAggregation',
    'MeanAggregation',
    'SumAggregation',
    'MaxAggregation',
    'MinAggregation',
    'MulAggregation',
    'VarAggregation',
    'StdAggregation',
    'SoftmaxAggregation',
    'PowerMeanAggregation',
    'LSTMAggregation',
    'Set2Set',
    'DegreeScalerAggregation',
    'SortAggr',
    'GraphMultisetTransformer',
    'AttentionalAggregation',
]

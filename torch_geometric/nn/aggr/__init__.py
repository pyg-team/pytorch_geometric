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
from .quantile import MedianAggregation, QuantileAggregation
from .lstm import LSTMAggregation
from .gru import GRUAggregation
from .set2set import Set2Set
from .scaler import DegreeScalerAggregation
from .equilibrium import EquilibriumAggregation
from .sort import SortAggregation
from .gmt import GraphMultisetTransformer
from .attention import AttentionalAggregation
from .mlp import MLPAggregation
from .deep_sets import DeepSetsAggregation
from .set_transformer import SetTransformerAggregation
from .lcm import LCMAggregation
from .variance_preserving import VariancePreservingAggregation
from .patch_transformer import PatchTransformerAggregation

__all__ = classes = [
    'Aggregation',
    'MultiAggregation',
    'SumAggregation',
    'MeanAggregation',
    'MaxAggregation',
    'MinAggregation',
    'MulAggregation',
    'VarAggregation',
    'StdAggregation',
    'SoftmaxAggregation',
    'PowerMeanAggregation',
    'MedianAggregation',
    'QuantileAggregation',
    'LSTMAggregation',
    'GRUAggregation',
    'Set2Set',
    'DegreeScalerAggregation',
    'SortAggregation',
    'GraphMultisetTransformer',
    'AttentionalAggregation',
    'EquilibriumAggregation',
    'MLPAggregation',
    'DeepSetsAggregation',
    'SetTransformerAggregation',
    'LCMAggregation',
    'VariancePreservingAggregation',
    'PatchTransformerAggregation',
]

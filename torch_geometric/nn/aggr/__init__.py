from .attention import AttentionalAggregation
from .base import Aggregation
from .basic import (MaxAggregation, MeanAggregation, MinAggregation,
                    MulAggregation, PowerMeanAggregation, SoftmaxAggregation,
                    StdAggregation, SumAggregation, VarAggregation)
from .deep_sets import DeepSetsAggregation
from .equilibrium import EquilibriumAggregation
from .gmt import GraphMultisetTransformer
from .gru import GRUAggregation
from .lcm import LCMAggregation
from .lstm import LSTMAggregation
from .mlp import MLPAggregation
from .multi import MultiAggregation
from .patch_transformer import PatchTransformerAggregation
from .quantile import MedianAggregation, QuantileAggregation
from .scaler import DegreeScalerAggregation
from .set2set import Set2Set
from .set_transformer import SetTransformerAggregation
from .sort import SortAggregation
from .variance_preserving import VariancePreservingAggregation

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

r"""Normalization package."""

from .batch_norm import BatchNorm, HeteroBatchNorm
from .diff_group_norm import DiffGroupNorm
from .graph_norm import GraphNorm
from .graph_size_norm import GraphSizeNorm
from .instance_norm import InstanceNorm
from .layer_norm import HeteroLayerNorm, LayerNorm
from .mean_subtraction_norm import MeanSubtractionNorm
from .msg_norm import MessageNorm
from .pair_norm import PairNorm

__all__ = [
    'BatchNorm',
    'HeteroBatchNorm',
    'InstanceNorm',
    'LayerNorm',
    'HeteroLayerNorm',
    'GraphNorm',
    'GraphSizeNorm',
    'PairNorm',
    'MeanSubtractionNorm',
    'MessageNorm',
    'DiffGroupNorm',
]

classes = __all__

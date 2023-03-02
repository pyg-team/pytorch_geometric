from .batch_norm import BatchNorm, HeteroBatchNorm
from .instance_norm import InstanceNorm
from .layer_norm import LayerNorm, HeteroLayerNorm
from .graph_norm import GraphNorm
from .graph_size_norm import GraphSizeNorm
from .pair_norm import PairNorm
from .mean_subtraction_norm import MeanSubtractionNorm
from .msg_norm import MessageNorm
from .diff_group_norm import DiffGroupNorm

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

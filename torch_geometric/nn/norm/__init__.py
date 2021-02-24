from .batch_norm import BatchNorm
from .instance_norm import InstanceNorm
from .layer_norm import LayerNorm
from .graph_size_norm import GraphSizeNorm
from .pair_norm import PairNorm
from .msg_norm import MessageNorm
from .diff_norm import DiffNorm

__all__ = [
    'BatchNorm',
    'InstanceNorm',
    'LayerNorm',
    'GraphSizeNorm',
    'PairNorm',
    'MessageNorm',
    'DiffNorm' 
]

classes = sorted(__all__)

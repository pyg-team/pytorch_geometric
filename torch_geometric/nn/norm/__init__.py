from .batch_norm import BatchNorm
from .instance_norm import InstanceNorm
from .layer_norm import LayerNorm
from .graph_size_norm import GraphSizeNorm
from .pair_norm import PairNorm
from .msg_norm import MessageNorm
from .diff_group_norm import DiffGroupNorm

__all__ = [
    'BatchNorm',
    'InstanceNorm',
    'LayerNorm',
    'GraphSizeNorm',
    'PairNorm',
    'MessageNorm',
    'DiffGroupNorm',
]

classes = __all__

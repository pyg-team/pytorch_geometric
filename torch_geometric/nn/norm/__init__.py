from .batch_norm import BatchNorm
from .diff_group_norm import DiffGroupNorm
from .graph_norm import GraphNorm
from .graph_size_norm import GraphSizeNorm
from .instance_norm import InstanceNorm
from .layer_norm import LayerNorm
from .msg_norm import MessageNorm
from .pair_norm import PairNorm

__all__ = [
    'BatchNorm',
    'InstanceNorm',
    'LayerNorm',
    'GraphNorm',
    'GraphSizeNorm',
    'PairNorm',
    'MessageNorm',
    'DiffGroupNorm',
]

classes = __all__

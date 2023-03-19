from .base import EdgeUpdater
from .basics import (
    Fill,
    Cache,
    GCNNorm
)
from .loop import (
    MaskEdges, RemoveSelfLoops, SegregateSelfLoops,
    AddMaskedSelfLoops, AddSelfLoops, AddRemainingSelfLoops
)
from .compose import Compose

__all__ = classes = [
    'EdgeUpdater',
    'Fill',
    'Cache',
    'GCNNorm',
    'Compose',
    'MaskEdges',
    'RemoveSelfLoops',
    'SegregateSelfLoops',
    'AddMaskedSelfLoops',
    'AddSelfLoops',
    'AddRemainingSelfLoops'
]

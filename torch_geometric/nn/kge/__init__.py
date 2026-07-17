r"""Knowledge Graph Embedding (KGE) package."""

from .base import KGEModel
from .complex import ComplEx
from .distmult import DistMult
from .rotate import RotatE
from .transe import TransE

__all__ = classes = [
    'KGEModel',
    'TransE',
    'ComplEx',
    'DistMult',
    'RotatE',
]

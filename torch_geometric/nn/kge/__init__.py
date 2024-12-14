r"""Knowledge Graph Embedding (KGE) package."""

from .base import KGEModel
from .transd import TransD
from .transe import TransE
from .complex import ComplEx
from .distmult import DistMult
from .rotate import RotatE

__all__ = classes = [
    'KGEModel',
    'TransE',
    'TransD',
    'ComplEx',
    'DistMult',
    'RotatE',
]

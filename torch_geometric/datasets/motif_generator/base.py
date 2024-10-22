from abc import ABC, abstractmethod
from typing import Any

from torch_geometric.data import Data
from torch_geometric.resolver import resolver


class MotifGenerator(ABC):
    r"""An abstract base class for generating a motif."""
    @abstractmethod
    def __call__(self) -> Data:
        r"""To be implemented by :class:`Motif` subclasses."""

    @staticmethod
    def resolve(query: Any, *args: Any, **kwargs: Any) -> 'MotifGenerator':
        import torch_geometric.datasets.motif_generator as _motif_generators
        motif_generators = [
            gen for gen in vars(_motif_generators).values()
            if isinstance(gen, type) and issubclass(gen, MotifGenerator)
        ]
        return resolver(motif_generators, {}, query, MotifGenerator, 'Motif',
                        *args, **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

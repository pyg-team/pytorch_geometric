from abc import ABC, abstractmethod
from typing import Any

from torch_geometric.data import Data
from torch_geometric.resolver import resolver


class GraphGenerator(ABC):
    r"""An abstract base class for generating synthetic graphs."""
    @abstractmethod
    def __call__(self) -> Data:
        r"""To be implemented by :class:`GraphGenerator` subclasses."""
        raise NotImplementedError

    @staticmethod
    def resolve(query: Any, *args: Any, **kwargs: Any) -> 'GraphGenerator':
        import torch_geometric.datasets.graph_generator as _graph_generators
        graph_generators = [
            gen for gen in vars(_graph_generators).values()
            if isinstance(gen, type) and issubclass(gen, GraphGenerator)
        ]
        return resolver(graph_generators, {}, query, GraphGenerator, 'Graph',
                        *args, **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from torch_geometric.nn.aggr import Aggregation, MultiAggregation

class Select(torch.nn.Module, ABC):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                **kwargs) -> torch.tensor:
        """Groups nodes into clusters and returns a vector of cluster indices."""
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)

class Connect(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, mapping: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns a coarsened `edge_index` and `edge_attr`"""
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__)

class Pooling(torch.nn.Module):
    def __init__(self, select:Select, reduce: Aggregation, connect:Connect):
        """A base class for pooling methods. Based on 
        `"Understanding Pooling in Graph Neural Networks"
        <https://arxiv.org/abs/1905.05178>`_.
        The pooling method is composed of
        three components: a `select` that maps nodes to cluters, `reduce` that defines
        how node features in a cluster are aggregated, and 
        `connect` that defines how the edges are updated."""
        super().__init__()
        self.select = select
        self.reduce = reduce
        self.connect = connect
    
    def reset_parameters(self):
        self.select.reset_parameters()
        self.reduce.reset_parameters()
        self.connect.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor,
                                   Optional[torch.Tensor], torch.Tensor]:
        mapping = self.select(x, edge_index, edge_attr, **kwargs)
        x = self.reduce(x, mapping, **kwargs)
        edge_index, edge_attr = self.connect(mapping, edge_index, edge_attr, **kwargs)

        return x, edge_index, edge_attr, mapping

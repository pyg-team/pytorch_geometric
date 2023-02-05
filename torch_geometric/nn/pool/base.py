from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.pool.connect import Connect
from torch_geometric.nn.pool.select import Select
from torch_geometric.utils.mixin import CastMixin


@dataclass
class PoolingOutput(CastMixin):
    r"""The pooling output of a :class:`torch_geometric.nn.pool.Pooling`
    module.

    Args:
        x (torch.Tensor): The pooled node features.
        edge_index (torch.Tensor): The coarsened edge indices.
        edge_attr (torch.Tensor, optional): The edge features of the coarsened
            graph. (default: :obj:`None`)
        batch (torch.Tensor, optional): The batch vector of the pooled nodes.
    """
    x: Tensor
    edge_index: Tensor
    edge_attr: Optional[Tensor] = None
    batch: Optional[Tensor] = None


class Pooling(torch.nn.Module):
    r"""A base class for pooling layers based on the
    `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/1905.05178>`_ paper.

    :class:`Pooling` decomposes a pooling layer into three components:

    #. :class:`Select` defines how input nodes map to supernodes.

    #. :class:`Reduce` defines how input node features are aggregated.

    #. :class:`Connect` decides how the supernodes are connected to each other.

    Args:
        select (Select): The node selection operator.
        reduce (Select): The node feature aggregation operator.
        connect (Connect): The edge connection operator.
    """
    def __init__(self, select: Select, reduce: Aggregation, connect: Connect):
        super().__init__()
        self.select = select
        self.reduce = reduce
        self.connect = connect

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.select.reset_parameters()
        self.reduce.reset_parameters()
        self.connect.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> PoolingOutput:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        cluster, num_clusters = self.select(x, edge_index, edge_attr, batch)
        x = self.reduce(x, cluster, dim_size=num_clusters)
        edge_index, edge_attr = self.connect(cluster, edge_index, edge_attr,
                                             batch)

        if batch is not None:
            batch = (torch.arange(num_clusters, device=x.device)).scatter_(
                0, cluster, batch)

        return PoolingOutput(x, edge_index, edge_attr, batch)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  select={self.select},\n'
                f'  reduce={self.reduce},\n'
                f'  connect={self.connect},\n'
                f')')

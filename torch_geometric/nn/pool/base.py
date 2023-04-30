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
        cluster (torch.Tensor, optional): The mapping from nodes to supernodes.
        edge_attr (torch.Tensor, optional): The edge features of the coarsened
            graph. (default: :obj:`None`)
        batch (torch.Tensor, optional): The batch vector of the pooled nodes.
    """
    x: Tensor
    edge_index: Tensor
    cluster: Tensor
    edge_attr: Optional[Tensor] = None
    batch: Optional[Tensor] = None


class Pooling(torch.nn.Module):
    r"""A base class for pooling layers based on the
    `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/2110.05292>`_ paper.

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
        **kwargs: Optional[dict],
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
            **kwargs (optional): Additional arguments passed to the
                :meth:`forward` methods of :obj:`select`, :obj:`reduce` and
                :obj:`connect`.
        """
        num_nodes = x.size(0)

        # SELECT: Map nodes to supernodes.
        select_kwargs = self.select.inspector.distribute('forward', kwargs)
        cluster, num_clusters = self.select(x, edge_index, edge_attr, batch,
                                            **select_kwargs)
        # Some nodes might not be part of any
        # cluster.
        dropped_nodes_mask = cluster == -1
        for key, value in kwargs.items():
            if isinstance(value, Tensor) and value.size(0) == x.size(0):
                kwargs[key] = value[~dropped_nodes_mask]

        # REDUCE: Compute new node features.
        reduce_kwargs = self.reduce.inspector.distribute('forward', kwargs)
        x = self.reduce(x[~dropped_nodes_mask], cluster[~dropped_nodes_mask],
                        ptr=None, dim_size=num_clusters, dim=0,
                        **reduce_kwargs)

        # CONNECT: Compute new edge indices.
        connect_kwargs = self.connect.inspector.distribute('forward', kwargs)
        edge_index, edge_attr = self.connect(cluster, edge_index, num_nodes,
                                             edge_attr, batch,
                                             **connect_kwargs)

        if batch is not None:
            batch = batch[~dropped_nodes_mask]
            batch = (torch.arange(num_clusters, device=x.device)).scatter_(
                0, cluster[~dropped_nodes_mask], batch)

        return PoolingOutput(x, edge_index, cluster, edge_attr, batch)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  select={self.select},\n'
                f'  reduce={self.reduce},\n'
                f'  connect={self.connect},\n'
                f')')

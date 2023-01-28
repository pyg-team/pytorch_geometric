from typing import Optional, Tuple

import torch

from torch_geometric.nn.aggr import Aggregation

from .connect import Connect
from .select import Select


class Pooling(torch.nn.Module):
    def __init__(self, select: Select, reduce: Aggregation, connect: Connect):
        """A base class for pooling methods. Based on
        `"Understanding Pooling in Graph Neural Networks"
        <https://arxiv.org/abs/1905.05178>`_. The pooling method is composed
        of three components: :obj:`select` maps nodes to super node (cluster),
        :obj:`reduce` defines how node features in a super node are aggregated,
        and :obj:`connect` that defines how the edges are updated.
        """
        super().__init__()
        self.select = select
        self.reduce = reduce
        self.connect = connect

    def reset_parameters(self):
        self.select.reset_parameters()
        self.reduce.reset_parameters()
        self.connect.reset_parameters()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               torch.Tensor]:
        mapping = self.select(x, edge_index, edge_attr, batch, **kwargs)
        x = self.reduce(x, mapping, **kwargs)
        edge_index, edge_attr = self.connect(mapping, edge_index, edge_attr,
                                             **kwargs)

        return x, edge_index, edge_attr, mapping

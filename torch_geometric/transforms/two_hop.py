import torch

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


@functional_transform('two_hop')
class TwoHop(BaseTransform):
    r"""Adds the two hop edges to the edge indices
    (functional name: :obj:`two_hop`)."""
    def forward(self, data: Data) -> Data:
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        if torch_geometric.typing.WITH_WINDOWS:
            adj = to_torch_coo_tensor(edge_index, size=(N, N))
        else:
            adj = to_torch_csr_tensor(edge_index, size=(N, N))

        adj = adj @ adj

        edge_index2, _ = to_edge_index(adj)
        edge_index2, _ = remove_self_loops(edge_index2)

        edge_index = torch.cat([edge_index, edge_index2], dim=1)

        if edge_attr is not None:
            # We treat newly added edge features as "zero-features":
            edge_attr2 = edge_attr.new_zeros(edge_index2.size(1),
                                             *edge_attr.size()[1:])
            edge_attr = torch.cat([edge_attr, edge_attr2], dim=0)

        data.edge_index, data.edge_attr = coalesce(edge_index, edge_attr, N)

        return data

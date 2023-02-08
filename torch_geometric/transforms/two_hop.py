import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, remove_self_loops


@functional_transform('two_hop')
class TwoHop(BaseTransform):
    r"""Adds the two hop edges to the edge indices
    (functional name: :obj:`two_hop`)."""
    def __call__(self, data: Data) -> Data:
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(N, N))

        adj = adj @ adj
        row, col, _ = adj.coo()
        edge_index2 = torch.stack([row, col], dim=0)
        edge_index2, _ = remove_self_loops(edge_index2)

        edge_index = torch.cat([edge_index, edge_index2], dim=1)
        if edge_attr is None:
            data.edge_index = coalesce(edge_index, num_nodes=N)
        else:
            # We treat newly added edge features as "zero-features":
            edge_attr2 = edge_attr.new_zeros(edge_index2.size(1),
                                             *edge_attr.size()[1:])
            edge_attr = torch.cat([edge_attr, edge_attr2], dim=0)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N)
            data.edge_index, data.edge_attr = edge_index, edge_attr

        return data

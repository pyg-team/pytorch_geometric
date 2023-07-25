import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, to_torch_csr_tensor, to_undirected


@functional_transform('two_hop')
class TwoHop(BaseTransform):
    r"""Adds the two hop edges to the edge indices
    (functional name: :obj:`two_hop`)."""
    def forward(self, data: Data) -> Data:
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        if data.is_directed():
            edge_index = to_undirected(edge_index, N)

        # Convert to torch_csr_tensor and perform matrix multiplication
        adj = to_torch_csr_tensor(edge_index, size=(N, N))
        edge_index2 = (adj @ adj).coalesce()

        # Remove self-loops and duplicate edges
        edge_index2 = edge_index2.remove_self_loops()

        # Combine edge_index and edge_index2
        edge_index_combined = torch.cat([edge_index, edge_index2.indices()], dim=1)

        # Combine edge_attr if it exists
        if edge_attr is not None:
            num_new_edges = edge_index2.indices().size(1)
            edge_attr2 = edge_attr.new_zeros(num_new_edges, *edge_attr.size()[1:])
            edge_attr_combined = torch.cat([edge_attr, edge_attr2], dim=0)

        # Update data object in-place with the new edge_index and edge_attr
        data.edge_index, data.edge_attr = coalesce(edge_index_combined, edge_attr_combined, N)

        return data

import torch
from torch_sparse import spmm
from torch_geometric.utils import degree, remove_self_loops, add_self_loops


class GCNProp(torch.nn.Module):
    def __init__(self):
        super(GCNProp, self).__init__()

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        num_nodes, num_edges = x.size(0), row.size(0)

        if edge_attr is None:
            edge_attr = x.new_ones((num_edges, ))
        assert edge_attr.dim() == 1

        # Normalize adjacency matrix.
        deg = degree(row, num_nodes, dtype=x.dtype, device=x.device)
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        edge_attr = deg[row] * edge_attr * deg[col]

        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_attr = torch.cat([edge_attr, deg * deg], dim=0)
        edge_index = add_self_loops(edge_index, num_nodes)

        # Perform the propagation.
        out = spmm(edge_index, edge_attr, num_nodes, x)

        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

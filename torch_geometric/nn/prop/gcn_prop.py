import torch
from torch_sparse import spmm
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, add_self_loops


class GCNProp(torch.nn.Module):
    def __init__(self, improved=False):
        super(GCNProp, self).__init__()
        self.improved = improved

    def forward(self, x, edge_index, edge_attr=None):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), ))
        assert edge_attr.dim() == 1

        # Add self-loops to adjacency matrix.
        edge_index = add_self_loops(edge_index, x.size(0))
        loop_value = x.new_full((x.size(0), ), 1 if not self.improved else 2)
        edge_attr = torch.cat([edge_attr, loop_value], dim=0)

        # Normalize adjacency matrix.
        row, col = edge_index
        deg = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        edge_attr = deg[row] * edge_attr * deg[col]

        # Perform the propagation.
        out = spmm(edge_index, edge_attr, x.size(0), x)

        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

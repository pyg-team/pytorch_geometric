import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, remove_self_loops, scatter


@functional_transform('line_graph')
class LineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-graph
    (functional name: :obj:`line_graph`):

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    For undirected graphs, the maximum line-graph node index is
    :obj:`(data.edge_index.size(1) // 2) - 1`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed: bool = False):
        self.force_directed = force_directed

    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is None:
            edge_index = coalesce(edge_index, num_nodes=N)
        else:
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N)
        row, col = edge_index

        if self.force_directed or data.is_directed():
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)

            cols = [
                i[cumsum[col[j]]:cumsum[col[j] + 1]]
                for j in range(col.size(0))
            ]
            rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]

            row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

            data.edge_index = torch.stack([row, col], dim=0)
            data.x = data.edge_attr
            data.num_nodes = edge_index.size(1)

        else:
            # Compute node indices.
            mask = row < col
            row, col = row[mask], col[mask]
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            (row, col), i = coalesce(
                torch.stack([
                    torch.cat([row, col], dim=0),
                    torch.cat([col, row], dim=0)
                ], dim=0),
                torch.cat([i, i], dim=0),
                N,
            )

            # Compute new edge indices according to `i`.
            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            joints = torch.split(i, count.tolist())

            def generate_grid(x):
                row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
                col = x.repeat(x.numel())
                return torch.stack([row, col], dim=0)

            joints = [generate_grid(joint) for joint in joints]
            joints = torch.cat(joints, dim=1)
            joints, _ = remove_self_loops(joints)
            N = row.size(0) // 2
            joints = coalesce(joints, num_nodes=N)

            if edge_attr is not None:
                data.x = scatter(edge_attr, i, dim=0, dim_size=N, reduce='sum')
            data.edge_index = joints
            data.num_nodes = edge_index.size(1) // 2

        data.edge_attr = None
        return data

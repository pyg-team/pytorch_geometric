import torch
from torch_sparse import coalesce
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops


class LineGraph(object):
    r"""Converts a graph to its corresponding line-graph:

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    For directed graphs, line graph vertex indices are equal to indices in the
    original graph's `edge_index`. For undirected graphs, line graph vertex
    indices are obtained by ordering `(row, col)` edges with `row < col`, so
    that the maximum line graph vertex index is
    `original.edge_index.size(1) // 2 - 1`.

    New node features are given by (scattered) old edge attributes: for
    undirected graphs, edge attributes for reciprocal edges `(row, col)` and
    `(col, row)` get summed together.

    If `force_directed` is `True`, the graph will be treated as a directed
    graph even if `data.is_directed() == False`.
    """

    def __call__(self, data, force_directed=False):
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N)

        if force_directed or data.is_directed():
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            count = scatter_add(
                torch.ones_like(row), row, dim=0, dim_size=data.num_nodes)
            cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)

            cols = [
                i[cumsum[col[j]]:cumsum[col[j] + 1]]
                for j in range(col.size(0))
            ]
            rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]

            row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

            data.edge_index = torch.stack([row, col], dim=0)
            data.x = data.edge_attr

        else:
            # Compute node indices.
            mask = row < col
            row, col = row[mask], col[mask]
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            (row, col), i = coalesce(
                torch.stack(
                    [
                        torch.cat([row, col], dim=0),
                        torch.cat([col, row], dim=0)
                    ],
                    dim=0), torch.cat([i, i], dim=0), N, N)

            # Compute new edge indices according to `i`.
            count = scatter_add(
                torch.ones_like(row), row, dim=0, dim_size=data.num_nodes)
            joints = torch.split(i, count.tolist())

            def generate_grid(x):
                row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
                col = x.repeat(x.numel())
                return torch.stack([row, col], dim=0)

            joints = [generate_grid(joint) for joint in joints]
            joints = torch.cat(joints, dim=1)
            joints, _ = remove_self_loops(joints)
            N = row.size(0) // 2
            joints, _ = coalesce(joints, None, N, N)

            if edge_attr is not None:
                print("DRIN")
                data.x = scatter_add(edge_attr, i, dim=0, dim_size=N)
            data.edge_index = joints

        data.edge_attr = None
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

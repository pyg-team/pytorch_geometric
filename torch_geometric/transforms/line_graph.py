import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    coalesce,
    cumsum,
    remove_self_loops,
    scatter,
    to_undirected,
)


@functional_transform('line_graph')
class LineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-graph
    (functional name: :obj:`line_graph`).

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &=
         \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset,\, e_1 \neq e_2 \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed: bool = False) -> None:
        self.force_directed = force_directed

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=N)
        row, col = edge_index

        def vector_arange(start: Tensor, step: Tensor) -> Tensor:
            step_csum = cumsum(step)[:-1]
            offset = (step_csum - start).repeat_interleave(step)
            rng = torch.arange(offset.size(0), dtype=torch.long,
                               device=offset.device)
            return rng - offset

        if self.force_directed or data.is_directed():
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            ptr = cumsum(count)
            col_count = count[col]
            row = i.repeat_interleave(col_count)
            col_index = vector_arange(ptr[col], col_count)
            col = i[col_index]

            data.edge_index = torch.stack([row, col], dim=0)
            data.x = data.edge_attr
            data.num_nodes = edge_index.size(1)

        else:
            # Compute node indices.
            mask = row <= col
            edge_index = edge_index[:, mask]
            N = edge_index.size(1)
            i = torch.arange(N, dtype=torch.long, device=row.device)
            (row, col), i = to_undirected(edge_index, i, num_nodes=N,
                                          reduce='min')

            # Compute new edge indices according to `i`.
            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            ptr = cumsum(count)
            row = i.repeat_interleave(count[row])
            count_rep = count.repeat_interleave(count)
            col_index = vector_arange(ptr[:-1].repeat_interleave(count),
                                      count_rep)
            col = i[col_index]

            index = torch.stack([row, col], dim=0)
            index, _ = remove_self_loops(index)
            index = coalesce(index, num_nodes=N)

            if edge_attr is not None:
                data.x = scatter(edge_attr, i, dim=0, dim_size=N, reduce='sum')
            data.edge_index = index
            data.num_nodes = N

        data.edge_attr = None

        return data

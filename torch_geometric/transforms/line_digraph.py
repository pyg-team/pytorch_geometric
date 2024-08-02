import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce


@functional_transform('line_digraph')
class LineDiGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-digraph
    (functional name: :obj:`line_digraph`).

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ ((u, v), (w, x)) : (u, v) \in \mathcal{E}
            \land (w, x) \in \mathcal{E} \land v = w\}

    Line-digraph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    """
    def __init__(self, force_directed: bool = False) -> None:
        self.force_directed = force_directed

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.is_directed()
        edge_index, edge_attr = data.edge_index, data.edge_attr
        E = data.num_edges

        edge_index, edge_attr = coalesce(edge_index, edge_attr, data.num_nodes)
        row, col = edge_index

        new_edge_index = []
        for i in range(E):
            new_col_i = torch.nonzero(row == col[i])
            new_row_i = i * torch.ones_like(new_col_i)
            new_edge_index.append(torch.cat([new_row_i, new_col_i], dim=1))
        new_edge_index = torch.cat(new_edge_index, dim=0).t()

        data.edge_index = new_edge_index
        data.x = edge_attr
        data.num_nodes = E
        data.edge_attr = None
        return data

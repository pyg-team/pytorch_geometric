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
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.is_directed()
        edge_index, edge_attr = data.edge_index, data.edge_attr
        E = data.num_edges

        edge_index, edge_attr = coalesce(edge_index, edge_attr, data.num_nodes)
        row, col = edge_index

        # Broadcast to create a mask for matching row and col elements
        mask = row.unsqueeze(0) == col.unsqueeze(1)  # (num_edges, num_edges)
        new_edge_index = torch.nonzero(mask).T

        data.edge_index = new_edge_index
        data.x = edge_attr
        data.num_nodes = E
        data.edge_attr = None
        return data

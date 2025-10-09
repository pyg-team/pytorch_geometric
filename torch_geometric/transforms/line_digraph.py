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

    Args:
        node_to_edge_features (str, optional): If set to :obj:`'none'`, the
            node attributes of the original graph will not be put into the line
            digraph. Otherwise, for each line digraph edge
            :obj:`e = ((u, v), (v, x))`, :obj:`'inner'` will use node
            attributes of :obj:`v` as edge attributes, :obj:`'outer'` will use
            node attributes of :obj:`u` and :obj:`x` as edge attributes, and
            :obj:`'all'` will use node attributes of :obj:`u`, :obj:`v` and
            :obj:`x` as edge attributes.
    """
    def __init__(self, node_to_edge_features: str = 'none'):
        assert node_to_edge_features in ['none', 'inner', 'outer', 'all']
        self.node_to_edge_features = node_to_edge_features

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.is_directed()
        edge_index, edge_attr = data.edge_index, data.edge_attr
        new_num_nodes = data.num_edges

        edge_index, edge_attr = coalesce(edge_index, edge_attr, data.num_nodes)
        row, col = edge_index

        # Broadcast to create a mask for matching row and col elements
        mask = row.unsqueeze(0) == col.unsqueeze(1)  # (num_edges, num_edges)
        new_edge_index = torch.nonzero(mask).T

        # Obtain new edge attributes
        if data.x is None or self.node_to_edge_features == 'none':
            new_edge_attr = None
        else:
            node_features = data.x
            edge_src = row[new_edge_index[0]]
            edge_mid = col[new_edge_index[0]]
            edge_dst = col[new_edge_index[1]]

            if self.node_to_edge_features == "inner":
                new_edge_attr = node_features[edge_mid]
            elif self.node_to_edge_features == "outer":
                new_edge_attr = torch.cat(
                    [node_features[edge_src], node_features[edge_dst]], dim=-1)
            else:  # self.node_to_edge_features == "all"
                new_edge_attr = torch.cat([
                    node_features[edge_src], node_features[edge_mid],
                    node_features[edge_dst]
                ], dim=-1)

        data.edge_index = new_edge_index
        data.x = edge_attr
        data.num_nodes = new_num_nodes
        data.edge_attr = new_edge_attr
        return data

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing


class CGCNNConv(MessagePassing):
    r"""Implements the convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`
    paper.

    .. math::

        \mathbf{x}^{\prime}_i = \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \Big(
        \sigma \( \mathbf{z}_{i, j} \mathbf{W}_f + \mathbf{b}_f \) \cdot
        g\( \mathbf{z}_{i, j} \mathbf{W}_s + \mathbf{b}_s  \) \Big),

    where :math:`\mathbf{z}_{i, j} =
        \[ \mathbf{x}_i, \mathbf{x}_j, \mathbf{e}_{i,j} \]`
    *.i.e.* concatenates features for neighboring vectors.
    """
    def __init__(self, node_dim, edge_dim, aggr="add", **kwargs):
        r"""
        Args:
            node_dim (int): Size of each node feature.
            edge_dim (int): Size of each edge feature.
            aggr (string, optional): The aggregation scheme to use
                (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
                (default: :obj:`"add"`)
        """
        super(CGCNNConv, self).__init__(aggr=aggr, **kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.fc_pre = nn.Sequential(
            nn.Linear(2 * self.node_dim + self.edge_dim, 2 * self.node_dim),
            nn.BatchNorm1d(2 * self.node_dim),
        )

        self.fc_post = nn.Sequential(nn.BatchNorm1d(self.node_dim))

    def forward(self, x, edge_index, edge_attr):
        r"""
        Args:
            x of shape [num_nodes, node_dim]
            edge_index of shape [2, num_edges]
            edge_attr of shape [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        r"""
        Args:
            x_i of shape [num_edges, node_dim]
            x_j of shape [num_edges, node_dim]
            edge_attr of shape [num_edges, edge_dim]

        Returns:
            tensor of shape [num_edges, node_dim]
        """
        z = self.fc_pre(torch.cat([x_i, x_j, edge_attr], dim=1))
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2

    def update(self, aggr_out, x):
        r"""
        Args:
            aggr_out of shape [num_nodes, node_dim]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` for CGCNN).
            x has shape [num_nodes, node_dim]

        Returns:
            tensor of shape [num_nodes, node_dim]
        """
        aggr_out = nn.Softplus()(x + self.fc_post(aggr_out))
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.node_dim,
                                   self.node_dim)

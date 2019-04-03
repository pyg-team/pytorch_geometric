import torch
from torch_geometric.nn.conv import MessagePassing

from ..inits import reset


class PointConv(MessagePassing):
    r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation"
    <https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ papers

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j \,
        \Vert \, \mathbf{p}_j - \mathbf{p}_i) \right),

    where :math:`\gamma_{\mathbf{\Theta}}` and
    :math:`h_{\mathbf{\Theta}}` denote neural
    networks, *.i.e.* MLPs, and :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    defines the position of each point.

    Args:
        local_nn (nn.Sequential, optional): Neural network
            :math:`h_{\mathbf{\Theta}}`. (default: :obj:`None`)
        global_nn (nn.Sequential, optional): Neural network
            :math:`\gamma_{\mathbf{\Theta}}`. (default: :obj:`None`)
    """

    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__('max')

        self.local_nn = local_nn
        self.global_nn = global_nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, pos, edge_index):
        r"""
        Args:
            x (Tensor): The node feature matrix Can be set to :obj:`None`.
            pos (Tensor or tuple): The node position matrix. Either given as
                tensor for use in general message passing or as tuple for use
                in message passing in bipartite graphs.
            edge_index (LongTensor): The edge indices.
        """
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_j, pos_i):
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def update(self, aggr_out):
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)

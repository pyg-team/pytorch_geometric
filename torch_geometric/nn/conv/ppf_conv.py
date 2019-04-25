import torch
from torch_geometric.nn.conv import MessagePassing
from ..inits import reset


def point_pair_features(pos_i, pos_j, norm_i, norm_j):
    def get_angle(v1, v2):
        return torch.atan2(
            torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))

    pseudo = pos_j - pos_i
    return torch.stack([
        pseudo.norm(p=2, dim=1),
        get_angle(norm_i, pseudo),
        get_angle(norm_j, pseudo),
        get_angle(norm_i, norm_j)
    ],
                       dim=1)


class PPFConv(MessagePassing):
    r"""The PPF-Net operator from the `"PPFNet: Global Context Aware Local
    Features for Robust 3D Point Matching" <https://arxiv.org/abs/1802.02669>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \Vert l \Vert
        \angle(n1, d) \Vert \angle(n2, d) \Vert \angle(n1,n2)  )\right)

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote neural networks, *.i.e.* MLPs,
    :math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
    each point, :math:`d' the vector defining the distance between two points,
    :math:`l' the normalized length of :math:`d', :math:`n1' and :math:`n2' the
    normal vectors of the points.
    The angle calculation is defined by :math:`\angle(v_1, v_2) =
    atan2(\lVert v_1 \times v_2 \rVert, v_1 \cdot v_2)'.

    Args:
        local_nn (nn.Sequential, optional): Neural network
            :math:`h_{\mathbf{\Theta}}`. (default: :obj:`None`)
        global_nn (nn.Sequential, optional): Neural network
            :math:`\gamma_{\mathbf{\Theta}}`. (default: :obj:`None`)
    """

    def __init__(self, local_nn=None, global_nn=None):
        super(PPFConv, self).__init__('max')

        self.local_nn = local_nn
        self.global_nn = global_nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, pos, norm, edge_index):
        r"""
        Args:
            x (Tensor): The node feature matrix. Allowed to be :obj:`None`.
            pos (Tensor or tuple): The node position matrix. Either given as
                tensor for use in general message passing or as tuple for use
                in message passing in bipartite graphs.
            norm (Tensor or tuple): The normal vectors of each node. Either
                given as tensor for use in general message passing or as tuple
                for use in message passing in bipartite graphs.
            edge_index (LongTensor): The edge indices.
        """
        return self.propagate(edge_index, x=x, pos=pos, norm=norm)

    def message(self, x_j, pos_i, pos_j, norm_i, norm_j):
        msg = point_pair_features(pos_i, pos_j, norm_i, norm_j)
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

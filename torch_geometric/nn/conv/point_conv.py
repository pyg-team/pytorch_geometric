import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, scatter_

from ..inits import reset


class PointConv(torch.nn.Module):
    r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation"
    <https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ papers

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j \,
        \Vert \, \mathbf{p}_j - \mathbf{p}_i) \right)

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
        super(PointConv, self).__init__()

        self.local_nn = local_nn
        self.global_nn = global_nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, pos, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        out = pos[col] - pos[row]

        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            out = torch.cat([x[col], out], dim=1)

        if self.local_nn is not None:
            out = self.local_nn(out)

        out, _ = scatter_('max', x, row, dim_size=pos.size(0))

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.dense.mincut_pool import _rank3_trace

EPS = 1e-15


class DMoNPooling(torch.nn.Module):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the learned cluster assignment matrix, the pooled node feature
    matrix, the coarsened symmetrically normalized adjacency matrix, and three
    auxiliary objectives: (1) The spectral loss

    .. math::
        \mathcal{L}_s = - \frac{1}{2m}
        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

    where :math:`\mathbf{B}` is the modularity matrix, (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F

    where :math:`C` is the number of clusters, and (3) the cluster loss

    .. math::
        \mathcal{L}_c = \frac{\sqrt{C}}{n}
        {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1.

    .. note::

        For an example of using :class:`DMoNPooling`, see
        `examples/proteins_dmon_pool.py
        <https://github.com/pyg-team/pytorch_geometric/blob
        /master/examples/proteins_dmon_pool.py>`_.

    Args:
        channels (int or List[int]): Size of each input sample. If given as a
            list, will construct an MLP based on the given feature sizes.
        k (int): The number of clusters.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(self, channels: Union[int, List[int]], k: int,
                 dropout: float = 0.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        from torch_geometric.nn.models.mlp import MLP
        self.mlp = MLP(channels + [k], act=None, norm=None)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), C = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool,
                              device=x.device)

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

        out = F.selu(torch.matmul(s.transpose(1, 2), x))
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        degrees = torch.einsum('ijk->ij', adj)  # B X N
        degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
        degrees_t = degrees.transpose(1, 2)  # B x 1 x N

        m = torch.einsum('ijk->i', degrees) / 2  # B
        m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

        ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
        cb = torch.matmul(degrees_t, s)  # B x 1 x C

        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        decompose = out_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = spectral_loss.mean()

        # Orthogonality regularization:
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(C).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = ortho_loss.mean()

        # Cluster loss:
        cluster_size = torch.einsum('ijk->ik', s)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1)
        cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
        cluster_loss = cluster_loss.mean()

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(C, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mlp.in_channels}, '
                f'num_clusters={self.mlp.out_channels})')

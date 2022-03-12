from typing import List, Union

import torch
import torch.nn.functional as F

EPS = 1e-15


class DMonPooling(torch.nn.Module):
    r"""Spectral modularity pooling operator from the `"Graph Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns learned cluster assignment matrix, pooled node feature matrix,
    coarsened symmetrically normalized adjacency matrix, and two auxiliary
    objectives: (1) The spectral loss

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

        For an example of using :class:`DMonPooling`, see
        `examples/proteins_dmon_pool.py
        <https://github.com/pyg-team/pytorch_geometric/blob
        /master/examples/proteins_dmon_pool.py>`_.

    Args:
        channels (List[int]): List of input and intermediate channels in order
            to construct an MLP.
        k (int): The number of clusters.
        dropout (float, optional): Dropout probability. (default: :obj:`0`)
    """
    def __init__(self, channels: Union[int, List[int]], k: int,
                 dropout: float = 0.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        from torch_geometric.nn.models.mlp import MLP
        self.mlp = MLP(channels + [k], act='selu', batch_norm=False)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x, adj, mask=None):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in
                \mathbb{R}^{B \times N \times F}` with batch-size
                :math:`B`, (maximum) number of nodes :math:`N` for each graph,
                and feature dimension :math:`F`. Since the cluster assignment
                matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
                is being created within this method, the MLP and softmax do
                not have to be applied beforehand.
            adj (Tensor): Symmetrically normalized adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
            :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
        """

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)

        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out = F.selu(out)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        degrees = torch.einsum('ijk->ik', adj).transpose(0, 1)
        m = torch.einsum('ij->', degrees)

        ca = torch.matmul(s.transpose(1, 2), degrees)
        cb = torch.matmul(degrees.transpose(0, 1), s)

        normalizer = torch.matmul(ca, cb) / 2 / m
        decompose = out_adj - normalizer
        spectral_loss = -self._rank3_trace(decompose) / 2 / m
        spectral_loss = torch.mean(spectral_loss)

        # Orthogonality regularization.
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        cluster_loss = torch.norm(torch.einsum(
            'ijk->ij', ss)) / adj.size(1) * torch.norm(i_s) - 1

        # Fix and normalize coarsened adjacency matrix.
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def _rank3_trace(self, x):
        return torch.einsum('ijj->i', x)

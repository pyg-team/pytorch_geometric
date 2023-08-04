import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def dense_asymcheegercut_pool(
        x: Tensor, adj: Tensor, s: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    The asymmetric cheeger cut pooling operator from the
    `"Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ paper.

    .. math::
        \mathbf{S} &= \text{softmax}(\mathbf{S}^{\prime})

        \mathbf{X}^{\prime} &= \mathbf{S}^{\top}
        \mathbf{X}

        \mathbf{A}^{\prime} &= \mathbf{S}^{\top}
        \mathbf{A} \mathbf{S}

    based on dense learned tensor :math:`\mathbf{S}^{\prime} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened adjacency matrix,
    and two auxiliary objectives:

    (1) The graph total variation loss

    .. math::
        \mathcal{L}_\text{GTV} = \frac{1}{2E} \sum_{k=1}^K \sum_{i=1}^N
        \sum_{j=i}^N a_{i,j} |s_{i,k} - s_{j,k}|,

    where :math:`E` is the number of edges/links.

    (2) The asymmetrical norm loss

    .. math::
        \mathcal{L}_\text{AN} = \frac{N(C - 1) -
        \sum_{k=1}^C ||\mathbf{s}_{:,k} -
        \textrm{quant}_{C-1} (\mathbf{s}_{:,k})||_{1, C-1}}{N(C-1)}

    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (torch.Tensor): Assignment tensor
            :math:`\mathbf{S}^{\prime} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied before-hand, since it is
            executed within this method.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    s = torch.softmax(s, dim=-1)

    batch_size, n_nodes, _ = x.size()

    if mask is not None:
        mask = mask.view(batch_size, n_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # Pooled features and adjacency
    out = torch.matmul(s.transpose(-2, -1), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(-2, -1), adj), s)

    # Total variation loss
    tv_loss = torch.mean(_totvar_loss(adj, s))

    # Balance loss
    bal_loss = torch.mean(_balance_loss(s))

    return out, out_adj, tv_loss, bal_loss


def _totvar_loss(adj, s):
    l1_norm = torch.sum(torch.abs(s[..., None, :] - s[:, None, ...]), dim=-1)
    loss = torch.sum(adj * l1_norm)

    # Normalize loss
    n_edges = torch.sum(adj != 0)
    loss *= 1 / (2 * n_edges)

    return loss


def _balance_loss(s):
    n_nodes, n_clust = s.size()[-2], s.size()[-1]

    # k-quantile
    idx = int(math.floor(n_nodes / n_clust))
    quant = torch.sort(s, dim=-2, descending=True)[0][:, idx, :]

    # Asymmetric l1-norm
    loss = s - torch.unsqueeze(quant, dim=1)
    loss = (loss >= 0) * (n_clust - 1) * loss + (loss < 0) * loss * -1
    loss = torch.sum(loss, dim=(-1, -2))  # shape [B]
    loss = 1 / (n_nodes * (n_clust - 1)) * (n_nodes * (n_clust - 1) - loss)

    return loss

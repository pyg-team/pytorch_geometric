from typing import Optional, Tuple

import torch
from torch import Tensor


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (torch.Tensor): Assignment tensor
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied before-hand, since it is
            executed within this method.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        temp (float, optional): Temperature parameter for softmax function.
            (default: :obj:`1.0`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    n_batches = s.shape[0]
    out_adj = torch.matmul(
        torch.stack(
            [torch.matmul(s.transpose(1, 2)[b], adj[b]) for b in range(n_batches)],
            dim=0,
        ),
        s,
    )
    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum("ijk->ij", adj)
    d = _sparse_rank3_diag(d_flat)
    # mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_den = _rank3_trace(
        torch.matmul(
            torch.stack(
                [torch.matmul(s.transpose(1, 2)[b], d[b]) for b in range(n_batches)],
                dim=0,
            ),
            s,
        )
    )

    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum("ijj->i", x)


def _sparse_diag_2d(diag):
    n = diag.shape[0]
    return torch.sparse_coo_tensor([list(range(n)), list(range(n))], diag)


def _sparse_rank3_diag(d_flat):
    return torch.stack([_sparse_diag_2d(diag.to_dense()) for diag in d_flat], dim=0)

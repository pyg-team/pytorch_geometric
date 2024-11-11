from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.dense.mincut_pool import _rank3_trace

EPS = 1e-15


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""MinCut pooling for sparse adjacency matrices.

    This class mirrors its dense counter-part in `nn.dense.mincut_pool`.

    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Sparse adjacency tensor
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

    num_clusters = s.size(-1)

    adj_bd = batched_block_diagonal_sparse(adj)
    s_bd = batched_block_diagonal_dense(s)

    # A' = S^T A S
    out_adj_bd = torch.sparse.mm(s_bd.T, torch.sparse.mm(adj_bd, s_bd))
    out_adj = block_diagonal_to_batched_3d(out_adj_bd, batch_size,
                                           num_clusters, num_clusters)

    # cut numerator, trace(S^T A S)
    mincut_num = _rank3_trace(out_adj)

    d_flat = torch.einsum("ijk->ij", adj)

    d = _sparse_rank3_diag(d_flat)
    d = batched_block_diagonal_sparse(d.coalesce())

    # cut denumerator, trace(S^T D S)
    mincut_den = _rank3_trace(
        block_diagonal_to_batched_3d(
            torch.sparse.mm(s_bd.T, torch.sparse.mm(d, s_bd)),
            batch_size,
            num_clusters,
            num_clusters,
        ))

    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def _sparse_diag_2d(diag: Tensor) -> Tensor:
    r"""Create a 2D sparse tensor given its diagonal values as input."""
    n = diag.shape[0]
    return torch.sparse_coo_tensor([list(range(n)), list(range(n))], diag)


def _sparse_rank3_diag(d_flat: Tensor) -> Tensor:
    r"""Construct batched 2D diagonal tensor,
    where the inner 2 dimensions corresponds to a sparse diagonal matrix.
    """
    return torch.stack([_sparse_diag_2d(diag.to_dense()) for diag in d_flat],
                       dim=0)


def batched_block_diagonal_sparse(batched_tensor_sp: Tensor) -> Tensor:
    """Convert a 3D batched sparse tensor
    into a 2D block diagonal structure.
    """
    if not batched_tensor_sp.is_sparse:
        raise TypeError("Expects a sparse tensor, but got a dense one")
    if batched_tensor_sp.layout != torch.sparse_coo:
        raise TypeError(
            f"Expects a sparse coo matrix, but got {batched_tensor_sp.layout}")

    if batched_tensor_sp.ndim != 3:
        raise TypeError(
            f"Expects dim to be 3, but got {batched_tensor_sp.ndim}")

    nnz = batched_tensor_sp._nnz()
    shape = batched_tensor_sp.size()
    b, h, w = shape
    indices = batched_tensor_sp.indices()
    # [b, h, w] -> [b*h, b*w]
    new_shape = (b * h, b * w)

    new_indices = torch.empty(2, nnz, device=indices.device,
                              dtype=indices.dtype)
    # indices: [b,h,w] -> [h+b*H, w+b*W]
    new_indices[0, :] = indices[1, :] + indices[0, :] * h
    new_indices[1, :] = indices[2, :] + indices[0, :] * w

    return torch.sparse_coo_tensor(indices=new_indices,
                                   values=batched_tensor_sp.values(),
                                   size=new_shape)


def batched_block_diagonal_dense(batched_tensor: Tensor) -> Tensor:
    """Convert a 3D batched dense tensor into a 2D block diagonal structure
    in sparse format.
    """
    if batched_tensor.is_sparse:
        raise TypeError("Expects a dense tensor, but got a sparse one")
    if batched_tensor.ndim != 3:
        raise TypeError(f"Expects dim to be 3, but got {batched_tensor.ndim}")

    shape = batched_tensor.size()
    b, h, w = shape
    nnz = b * h * w
    # [b, h, w] -> [b*h, b*w]
    new_shape = (b * h, b * w)

    new_indices = torch.empty(2, nnz, device=batched_tensor.device,
                              dtype=torch.int64)
    # indices: [b,h,w] -> [h+b*H, w+b*W]
    indices0 = torch.arange(0, b).repeat_interleave(h * w)
    indices1 = torch.arange(0, h).repeat(w, b).T.flatten()
    indices2 = torch.arange(0, w).repeat(h * b)
    new_indices[0, :] = indices1 + indices0 * h
    new_indices[1, :] = indices2 + indices0 * w

    return torch.sparse_coo_tensor(indices=new_indices,
                                   values=batched_tensor.flatten(),
                                   size=new_shape)


def block_diagonal_to_batched_3d(block_diagonal: Tensor, b: int, h: int,
                                 w: int) -> Tensor:
    """Convert a 2D block diagonal sparse matrix to 3D batched sparse one.

    b, h, w: dim0, dim1, dim2.
    """
    if not block_diagonal.is_sparse:
        raise TypeError("Expects a sparse tensor, but got a dense one")
    if block_diagonal.layout != torch.sparse_coo:
        raise TypeError(
            f"Expects a sparse coo matrix, but got {block_diagonal.layout}")
    return block_diagonal.values().reshape(b, h, w)

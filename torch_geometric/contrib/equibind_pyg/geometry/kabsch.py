"""
Kabsch rigid alignment utilities.

Implements the Kabsch algorithm to compute the optimal rotation and
translation aligning one point cloud to another in 3D, and helpers
to apply and compose these transforms. This functionality is used by
the EquiBind-Rigid model to convert predicted keypoints into global
SE(3) transforms, but is written generically for reuse.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _ensure_batch(points: Tensor) -> Tuple[Tensor, bool]:
    """
    Ensure input points have shape [B, N, 3].

    Args:
        points: Tensor of shape [N, 3] or [B, N, 3].

    Returns:
        A tuple (points_batched, added_batch_dim), where `points_batched`
        has shape [B, N, 3] and `added_batch_dim` indicates whether an
        artificial batch dimension was added.
    """
    if points.dim() == 2:
        # [N, 3] -> [1, N, 3]
        return points.unsqueeze(0), True
    elif points.dim() == 3:
        return points, False
    else:
        raise ValueError(f"Expected [N, 3] or [B, N, 3], got {points.shape}")


def kabsch(src: Tensor, tgt: Tensor, eps: float = 1e-7) -> Tuple[Tensor, Tensor]:
    """
    Compute the optimal rotation R and translation t (Kabsch algorithm)
    that aligns `src` onto `tgt` in the least-squares sense.

    Convention:

        aligned_src = src @ R.transpose(-1, -2) + t.unsqueeze(-2)

    such that aligned_src ≈ tgt.

    Args:
        src: [N, 3] or [B, N, 3]
            Source point sets (e.g., predicted coordinates).
        tgt: [N, 3] or [B, N, 3]
            Target point sets (e.g., ground-truth coordinates).
        eps: Small constant to avoid numerical issues (currently unused).

    Returns:
        R: [B, 3, 3] rotation matrices.
        t: [B, 3] translation vectors.
    """
    src_b, src_was_2d = _ensure_batch(src)
    tgt_b, tgt_was_2d = _ensure_batch(tgt)

    if src_b.shape != tgt_b.shape:
        raise ValueError(
            f"src and tgt shapes must match after batching, got {src_b.shape} vs {tgt_b.shape}"
        )

    B, N, _ = src_b.shape

    # Compute centroids
    src_centroid = src_b.mean(dim=-2, keepdim=True)  # [B, 1, 3]
    tgt_centroid = tgt_b.mean(dim=-2, keepdim=True)  # [B, 1, 3]

    # Center the point clouds
    src_centered = src_b - src_centroid  # [B, N, 3]
    tgt_centered = tgt_b - tgt_centroid  # [B, N, 3]

    # Covariance matrix H = src_centered^T @ tgt_centered
    H = torch.matmul(src_centered.transpose(-1, -2), tgt_centered)  # [B, 3, 3]

    # SVD: H = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)     # [B, 3, 3]
    Ut = U.transpose(-1, -2)     # [B, 3, 3]

    # Rotation: R = V @ U^T
    R = torch.matmul(V, Ut)      # [B, 3, 3]

    # Enforce det(R) = +1 (proper rotation)
    det_R = torch.det(R)         # [B]
    mask = det_R < 0
    if mask.any():
        # Flip last column of V for those batches
        V_corrected = V.clone()
        V_corrected[mask, :, -1] *= -1.0
        R = torch.matmul(V_corrected, Ut)

    # Translation: tgt_centroid = src_centroid @ R^T + t
    Rt = R.transpose(-1, -2)                            # [B, 3, 3]
    t = (tgt_centroid - torch.matmul(src_centroid, Rt)) # [B, 1, 3]
    t = t.squeeze(-2)                                   # [B, 3]

    return R, t


def apply_transform(points: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """
    Apply a rigid transformation defined by (R, t) to a set of points.

    Uses the same convention as `kabsch`:

        transformed = points @ R.transpose(-1, -2) + t.unsqueeze(-2)

    Args:
        points: [N, 3] or [B, N, 3].
        R:      [3, 3] or [B, 3, 3].
        t:      [3] or [B, 3].

    Returns:
        Transformed points with the same batch structure as `points`.
    """
    pts_b, pts_was_2d = _ensure_batch(points)

    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    if pts_b.size(0) != R.size(0) or R.size(0) != t.size(0):
        raise ValueError(
            f"Batch sizes must match: points {pts_b.size(0)}, R {R.size(0)}, t {t.size(0)}"
        )

    transformed = torch.matmul(pts_b, R.transpose(-1, -2)) + t.unsqueeze(-2)  # [B, N, 3]

    if pts_was_2d:
        return transformed.squeeze(0)
    return transformed


def kabsch_align(src: Tensor, tgt: Tensor) -> Tensor:
    """
    Align `src` to `tgt` using the Kabsch algorithm and return the aligned
    source point sets.

    Args:
        src: [N, 3] or [B, N, 3].
        tgt: [N, 3] or [B, N, 3].

    Returns:
        Tensor of aligned source coordinates with the same shape as `src`.
    """
    R, t = kabsch(src, tgt)
    return apply_transform(src, R, t)

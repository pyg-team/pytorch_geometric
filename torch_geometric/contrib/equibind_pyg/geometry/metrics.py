"""
Geometric metrics for evaluating docking poses.

Provides utilities to compute:

- RMSD between coordinate sets
- centroid distances between point clouds
- Kabsch-aligned RMSD

These are used both for training (as loss components) and evaluation
of pose quality in the EquiBind-Rigid model, but are written generically
for reuse in other 3D PyG models.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .kabsch import _ensure_batch, kabsch_align


def rmsd(coords1: Tensor, coords2: Tensor) -> Tensor:
    """
    Compute the per-batch RMSD between two sets of coordinates.

    Args:
        coords1: [N, 3] or [B, N, 3].
        coords2: [N, 3] or [B, N, 3].

    Returns:
        Scalar tensor if unbatched, or a [B] tensor if batched.
    """
    c1, was_2d_1 = _ensure_batch(coords1)
    c2, was_2d_2 = _ensure_batch(coords2)

    if c1.shape != c2.shape:
        raise ValueError(f"coords1 and coords2 must have same shape, got {c1.shape} vs {c2.shape}")

    diff = c1 - c2                                 # [B, N, 3]
    mse = (diff ** 2).sum(dim=-1).mean(dim=-1)     # [B]
    val = torch.sqrt(mse + 1e-12)                  # [B]

    if was_2d_1 and was_2d_2:
        return val.squeeze(0)
    return val


def centroid_distance(coords1: Tensor, coords2: Tensor) -> Tensor:
    """
    Compute the Euclidean distance between centroids of two point sets.

    Args:
        coords1: [N, 3] or [B, N, 3].
        coords2: [N, 3] or [B, N, 3].

    Returns:
        Scalar tensor if unbatched, or a [B] tensor if batched.
    """
    c1, was_2d_1 = _ensure_batch(coords1)
    c2, was_2d_2 = _ensure_batch(coords2)

    if c1.shape != c2.shape:
        raise ValueError(f"coords1 and coords2 must have same shape, got {c1.shape} vs {c2.shape}")

    centroid1 = c1.mean(dim=-2)   # [B, 3]
    centroid2 = c2.mean(dim=-2)   # [B, 3]

    diff = centroid1 - centroid2  # [B, 3]
    dist = torch.linalg.norm(diff, dim=-1)  # [B]

    if was_2d_1 and was_2d_2:
        return dist.squeeze(0)
    return dist


def kabsch_rmsd(src: Tensor, tgt: Tensor) -> Tensor:
    """
    Compute RMSD between `src` and `tgt` after optimal Kabsch alignment.

    Args:
        src: [N, 3] or [B, N, 3].
        tgt: [N, 3] or [B, N, 3].

    Returns:
        Scalar tensor if unbatched, or a [B] tensor if batched.
    """
    aligned = kabsch_align(src, tgt)
    return rmsd(aligned, tgt)

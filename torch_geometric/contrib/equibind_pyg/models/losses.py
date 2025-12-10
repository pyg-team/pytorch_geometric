"""
Loss functions for the EquiBind-Rigid model.

Defines a simple coordinate-regression objective combining:

- raw ligand RMSD
- Kabsch-aligned RMSD
- centroid distance

This provides a pragmatic rigid-docking loss that approximates parts
of the original EquiBind objective while omitting OT, intersection,
and torsion losses for simplicity.
"""

from __future__ import annotations

import torch
from torch import Tensor

from equibind_pyg.geometry.metrics import (
    rmsd,
    kabsch_rmsd,
    centroid_distance,
)


def compute_loss(
    out: dict,
    data,
    lambda_rmsd: float = 1.0,
    lambda_kabsch: float = 1.0,
    lambda_centroid: float = 0.1,
) -> dict:
    """
    Compute training losses for a rigid docking prediction.

    Args:
        out: Dict returned by `EquiBindRigid`, containing:
             - "ligand_pos_pred": [N_l, 3]
        data: Input HeteroData containing:
             - data.ligand_pos_bound: [N_l, 3] (ground-truth pose)
        lambda_rmsd:    Weight for raw RMSD term.
        lambda_kabsch:  Weight for Kabsch-aligned RMSD term.
        lambda_centroid:Weight for centroid distance term.

    Returns:
        Dictionary with the following keys:
            - "total":        Scalar total loss.
            - "rmsd":         Scalar RMSD loss.
            - "kabsch_rmsd":  Scalar Kabsch RMSD loss.
            - "centroid":     Scalar centroid-distance loss.
    """
    assert hasattr(data, "ligand_pos_bound"), (
        "HeteroData must contain ground truth coordinates in data.ligand_pos_bound"
    )

    pred = out["ligand_pos_pred"]          # [N_l, 3]
    target = data.ligand_pos_bound         # [N_l, 3]

    rmsd_loss = rmsd(pred, target)
    kabsch_loss = kabsch_rmsd(pred, target)
    centroid_loss = centroid_distance(pred, target)

    total = (
        lambda_rmsd * rmsd_loss
        + lambda_kabsch * kabsch_loss
        + lambda_centroid * centroid_loss
    )

    return {
        "total": total,
        "rmsd": rmsd_loss,
        "kabsch_rmsd": kabsch_loss,
        "centroid": centroid_loss,
    }

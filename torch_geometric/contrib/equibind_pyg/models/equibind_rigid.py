from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn, Tensor

from equibind_pyg.layers.iegnn import IEGNNLayer
from equibind_pyg.layers.keypoint_attention import KeypointAttention
from equibind_pyg.geometry.kabsch import kabsch, apply_transform


class EquiBindRigid(nn.Module):
    r"""
    Rigid EquiBind model in PyTorch Geometric.

    High-level pipeline:
        1. Run L IEGNN layers over ligand + receptor graphs.
        2. Extract K keypoints from ligand and receptor via learned queries.
        3. Use the Kabsch algorithm to compute an SE(3) transform aligning
           ligand keypoints to receptor keypoints.
        4. Apply the resulting rigid transform to ligand atoms to obtain
           the predicted pose.

    This implementation focuses on the rigid-body component and does not
    include flexible torsion modeling, OT-based keypoint losses, or
    collision/intersection losses; these can be added on top of this core
    architecture.

    Args:
        node_dim:       Node embedding dimension.
        num_layers:     Number of stacked IEGNN layers.
        num_keypoints:  Number of keypoints K for alignment.
        edge_mlp_dim:   Hidden size of EGNN edge MLPs.
        coord_mlp_dim:  Hidden size of EGNN coordinate MLPs.
        cross_attn_dim: Hidden size for cross-attention projections.
        lig_in_dim:     Optional input feature dimension for ligand nodes.
                        If provided and different from node_dim, a Linear
                        embedding layer is created in __init__.
        rec_in_dim:     Optional input feature dimension for receptor nodes.
                        If provided and different from node_dim, a Linear
                        embedding layer is created in __init__.
    """

    def __init__(
        self,
        node_dim: int = 128,
        num_layers: int = 4,
        num_keypoints: int = 32,
        edge_mlp_dim: int = 64,
        coord_mlp_dim: int = 64,
        cross_attn_dim: int = 64,
        lig_in_dim: Optional[int] = None,
        rec_in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.num_layers = num_layers
        self.num_keypoints = num_keypoints

        # If input feature dims are known (e.g. 28 for ligand, 21 for receptor)
        # we create embedding layers up front so that their weights can be
        # saved and restored cleanly with the checkpoint.
        self.lig_in_dim = lig_in_dim
        self.rec_in_dim = rec_in_dim

        if lig_in_dim is not None and lig_in_dim != node_dim:
            self.lig_embed: nn.Linear | None = nn.Linear(lig_in_dim, node_dim)
        else:
            self.lig_embed = None  # fallback: may be created lazily if needed

        if rec_in_dim is not None and rec_in_dim != node_dim:
            self.rec_embed: nn.Linear | None = nn.Linear(rec_in_dim, node_dim)
        else:
            self.rec_embed = None  # fallback: may be created lazily if needed

        # Stacked IEGNN layers
        self.layers = nn.ModuleList([
            IEGNNLayer(
                node_dim=node_dim,
                edge_mlp_dim=edge_mlp_dim,
                coord_mlp_dim=coord_mlp_dim,
                cross_attn_dim=cross_attn_dim,
            )
            for _ in range(num_layers)
        ])

        # Keypoint attention
        self.kp_lig = KeypointAttention(
            num_keypoints=num_keypoints,
            node_dim=node_dim,
            attn_dim=node_dim,
            out_dim=node_dim,
        )
        self.kp_rec = KeypointAttention(
            num_keypoints=num_keypoints,
            node_dim=node_dim,
            attn_dim=node_dim,
            out_dim=node_dim,
        )

    def forward(self, data) -> Dict[str, Tensor]:
        r"""
        Forward pass for a batch of heterogeneous protein–ligand graphs.

        Args:
            data: A `HeteroData` object containing:
                - data["ligand"].x          [N_l, F_l]
                - data["ligand"].pos        [N_l, 3]
                - data["ligand","intra","ligand"].edge_index
                - data["receptor"].x        [N_r, F_r]
                - data["receptor"].pos      [N_r, 3]
                - data["receptor","intra","receptor"].edge_index

        Returns:
            A dict with:
                - "ligand_pos_pred": [N_l, 3]
                - "ligand_x":        [N_l, node_dim]
                - "receptor_x":      [N_r, node_dim]
                - "ligand_kp_pos":   [K, 3]
                - "receptor_kp_pos": [K, 3]
                - "R":               [3, 3] rotation matrix
                - "t":               [3]    translation vector
        """
        # 1) Extract ligand + receptor from HeteroData
        lig_x = data["ligand"].x
        lig_pos = data["ligand"].pos
        lig_edge_index = data["ligand", "intra", "ligand"].edge_index

        rec_x = data["receptor"].x
        rec_pos = data["receptor"].pos
        rec_edge_index = data["receptor", "intra", "receptor"].edge_index

        # 2) Embed raw features to node_dim if needed

        # Ligand embedding:
        if self.lig_embed is not None:
            # Use pre-defined embedding layer
            lig_x = self.lig_embed(lig_x)
        elif lig_x.size(-1) != self.node_dim:
            # Fallback for synthetic/test data where alloc not done in __init__
            self.lig_embed = nn.Linear(lig_x.size(-1), self.node_dim).to(lig_x.device)
            lig_x = self.lig_embed(lig_x)

        # Receptor embedding:
        if self.rec_embed is not None:
            rec_x = self.rec_embed(rec_x)
        elif rec_x.size(-1) != self.node_dim:
            # Fallback for synthetic/test data
            self.rec_embed = nn.Linear(rec_x.size(-1), self.node_dim).to(rec_x.device)
            rec_x = self.rec_embed(rec_x)

        # 3) Stacked IEGNN layers
        for layer in self.layers:
            lig_x, lig_pos, rec_x, rec_pos = layer(
                lig_x,
                lig_pos,
                lig_edge_index,
                rec_x,
                rec_pos,
                rec_edge_index,
            )

        # 4) Keypoint extraction
        lig_kp_pos, lig_kp_feat, lig_attn = self.kp_lig(lig_x, lig_pos)
        rec_kp_pos, rec_kp_feat, rec_attn = self.kp_rec(rec_x, rec_pos)

        # 5) Align ligand keypoints to receptor keypoints using Kabsch
        R, t = kabsch(lig_kp_pos.unsqueeze(0), rec_kp_pos.unsqueeze(0))
        R = R.squeeze(0)
        t = t.squeeze(0)

        # 6) Apply transform to every ligand atom
        lig_pos_pred = lig_pos @ R.T + t

        return {
            "ligand_pos_pred": lig_pos_pred,
            "ligand_x": lig_x,
            "receptor_x": rec_x,
            "ligand_kp_pos": lig_kp_pos,
            "receptor_kp_pos": rec_kp_pos,
            "R": R,
            "t": t,
        }

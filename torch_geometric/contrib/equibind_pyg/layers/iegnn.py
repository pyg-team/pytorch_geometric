from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn, Tensor

from .egnn import EGNNLayer


class IEGNNLayer(nn.Module):
    r"""
    Interaction-EGNN (IEGNN) layer over a pair of graphs: ligand and receptor.

    This block applies:

      - An EGNN update on the ligand graph.
      - An EGNN update on the receptor graph.
      - Cross-graph attention from receptor -> ligand and ligand -> receptor
        on node features, prior to the EGNN updates.

    SE(3) properties:
      - Cross-graph attention uses only node features, not coordinates.
      - Coordinate updates are handled by EGNNLayer, which is SE(3)-equivariant.
      - Global rotations/translations of (lig_pos, rec_pos) are handled
        equivariantly; node features remain invariant.

    Args:
        node_dim:       Feature dimension for both ligand and receptor nodes.
        edge_mlp_dim:   Hidden dimension for EGNN edge MLP.
        coord_mlp_dim:  Hidden dimension for EGNN coordinate MLP.
        cross_attn_dim: Dimension for cross-attention queries/keys.
        aggr:           Aggregation method for EGNN ("mean", "sum", etc.).
    """

    def __init__(
        self,
        node_dim: int,
        edge_mlp_dim: int = 64,
        coord_mlp_dim: int = 64,
        cross_attn_dim: int = 64,
        aggr: str = "mean",
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.cross_attn_dim = cross_attn_dim

        # EGNN layers for each graph
        self.lig_egnn = EGNNLayer(
            in_dim=node_dim,
            out_dim=node_dim,
            edge_mlp_dim=edge_mlp_dim,
            coord_mlp_dim=coord_mlp_dim,
            aggr=aggr,
        )
        self.rec_egnn = EGNNLayer(
            in_dim=node_dim,
            out_dim=node_dim,
            edge_mlp_dim=edge_mlp_dim,
            coord_mlp_dim=coord_mlp_dim,
            aggr=aggr,
        )

        # Cross attention: receptor -> ligand
        self.lig_q = nn.Linear(node_dim, cross_attn_dim)
        self.rec_k = nn.Linear(node_dim, cross_attn_dim)
        self.rec_v = nn.Linear(node_dim, node_dim)

        # Cross attention: ligand -> receptor
        self.rec_q = nn.Linear(node_dim, cross_attn_dim)
        self.lig_k = nn.Linear(node_dim, cross_attn_dim)
        self.lig_v = nn.Linear(node_dim, node_dim)

        # Node updates after injecting cross-graph context
        self.lig_cross_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.rec_cross_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

    # --------------------------------------------------------------------- #
    # Cross-graph attention helpers
    # --------------------------------------------------------------------- #

    def _cross_attend(
        self,
        q_x: Tensor,
        k_x: Tensor,
        v_x: Tensor,
    ) -> Tensor:
        r"""
        Single-headed dot-product attention:

            attn = softmax( Q K^T / sqrt(d_k) )
            ctx  = attn V

        Args:
            q_x: [N_q, d_q] projected queries.
            k_x: [N_k, d_q] projected keys.
            v_x: [N_k, d_v] projected values.

        Returns:
            ctx: [N_q, d_v] context for each query node.
        """
        Q = q_x
        K = k_x
        V = v_x

        d_k = Q.size(-1)

        scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)  # [N_q, N_k]
        attn = torch.softmax(scores, dim=-1)               # [N_q, N_k]
        ctx = attn @ V                                     # [N_q, d_v]
        return ctx

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        lig_x: Tensor,
        lig_pos: Tensor,
        lig_edge_index: Tensor,
        rec_x: Tensor,
        rec_pos: Tensor,
        rec_edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            lig_x:          Ligand node features [N_l, node_dim].
            lig_pos:        Ligand coordinates [N_l, 3].
            lig_edge_index: Ligand intra-graph edges [2, E_l].

            rec_x:          Receptor node features [N_r, node_dim].
            rec_pos:        Receptor coordinates [N_r, 3].
            rec_edge_index: Receptor intra-graph edges [2, E_r].

        Returns:
            lig_x_out:  Updated ligand features [N_l, node_dim].
            lig_pos_out: Updated ligand coordinates [N_l, 3].
            rec_x_out:  Updated receptor features [N_r, node_dim].
            rec_pos_out: Updated receptor coordinates [N_r, 3].
        """
        # Cross-graph messages: receptor -> ligand
        lig_Q = self.lig_q(lig_x)  # [N_l, cross_attn_dim]
        rec_K = self.rec_k(rec_x)  # [N_r, cross_attn_dim]
        rec_V = self.rec_v(rec_x)  # [N_r, node_dim]

        lig_ctx_from_rec = self._cross_attend(lig_Q, rec_K, rec_V)  # [N_l, node_dim]

        # Cross-graph messages: ligand -> receptor
        rec_Q = self.rec_q(rec_x)  # [N_r, cross_attn_dim]
        lig_K = self.lig_k(lig_x)  # [N_l, cross_attn_dim]
        lig_V = self.lig_v(lig_x)  # [N_l, node_dim]

        rec_ctx_from_lig = self._cross_attend(rec_Q, lig_K, lig_V)  # [N_r, node_dim]

        # Fuse cross-graph context into node features
        lig_concat = torch.cat([lig_x, lig_ctx_from_rec], dim=-1)  # [N_l, 2 * node_dim]
        rec_concat = torch.cat([rec_x, rec_ctx_from_lig], dim=-1)  # [N_r, 2 * node_dim]

        lig_x_updated = self.lig_cross_mlp(lig_concat)             # [N_l, node_dim]
        rec_x_updated = self.rec_cross_mlp(rec_concat)             # [N_r, node_dim]

        # EGNN updates for each graph (features + coordinates)
        lig_x_out, lig_pos_out = self.lig_egnn(
            lig_x_updated,
            lig_pos,
            lig_edge_index,
        )

        rec_x_out, rec_pos_out = self.rec_egnn(
            rec_x_updated,
            rec_pos,
            rec_edge_index,
        )

        return lig_x_out, lig_pos_out, rec_x_out, rec_pos_out

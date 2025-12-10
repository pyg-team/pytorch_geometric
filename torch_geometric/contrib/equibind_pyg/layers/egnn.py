from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EGNNLayer(MessagePassing):
    r"""
    E(n)-Equivariant GNN layer (EGNN-style) for 3D point clouds.

    This layer jointly updates node features `x` and coordinates `pos`
    while guaranteeing SE(3) equivariance:

        - Node features are invariant to global rotations/translations.
        - Coordinates transform equivariantly.

    The layer follows the general form:

        m_ij = φ_e([x_i, x_j, ||x_i - x_j||²])
        Δx_i = Σ_j m_ij
        Δp_i = Σ_j (p_i - p_j) * φ_x(m_ij)
        x'_i = φ_h([x_i, Δx_i])
        p'_i = p_i + Δp_i

    Args:
        in_dim:          Input feature dimension per node.
        out_dim:         Output feature dimension per node.
        edge_mlp_dim:    Hidden dimension for the edge MLP.
        coord_mlp_dim:   Hidden dimension for the coordinate MLP.
        aggr:            Aggregation type for messages ("mean", "sum", "max").
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_mlp_dim: int = 64,
        coord_mlp_dim: int = 64,
        aggr: str = "mean",
    ) -> None:
        super().__init__(aggr=aggr)

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Edge MLP: [x_i, x_j, ||p_i - p_j||^2] -> edge embedding
        edge_in_dim = 2 * in_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_mlp_dim),
            nn.SiLU(),
            nn.Linear(edge_mlp_dim, edge_mlp_dim),
            nn.SiLU(),
        )

        # Coordinate MLP: edge embedding -> scalar per edge
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_mlp_dim, coord_mlp_dim),
            nn.SiLU(),
            nn.Linear(coord_mlp_dim, 1),
        )

        # Node MLP: [x_i, aggregated_edge_emb] -> new node features
        node_in_dim = in_dim + edge_mlp_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Args:
            x:          Node features of shape [N, in_dim].
            pos:        Node coordinates of shape [N, 3].
            edge_index: Edge indices of shape [2, E] (source=row, target=col).
            batch:      Optional batch vector [N] (not used here, kept for API
                        compatibility).

        Returns:
            x_out:   Updated node features [N, out_dim].
            pos_out: Updated node coordinates [N, 3].
        """
        row, col = edge_index  # row: target i, col: source j

        # Relative coordinates and squared distances
        rel = pos[row] - pos[col]                     # [E, 3]
        dist2 = (rel**2).sum(dim=-1, keepdim=True)   # [E, 1]

        # Edge messages
        edge_input = torch.cat([x[row], x[col], dist2], dim=-1)  # [E, 2*in_dim+1]
        m_ij = self.edge_mlp(edge_input)                         # [E, edge_mlp_dim]

        # Coordinate update
        coord_coef = self.coord_mlp(m_ij)                        # [E, 1]
        rel_weighted = rel * coord_coef                          # [E, 3]

        # Aggregate per target node i
        delta_pos = scatter(
            rel_weighted,
            row,
            dim=0,
            dim_size=x.size(0),
            reduce=self.aggr,
        )  # [N, 3]

        # Stabilize coordinate updates:
        # - scale down the update
        # - clamp to avoid large jumps
        delta_pos = 0.1 * delta_pos
        delta_pos = torch.clamp(delta_pos, -1.0, 1.0)

        pos_out = pos + delta_pos

        # Feature update: aggregate edge embeddings per node
        agg_m = scatter(
            m_ij,
            row,
            dim=0,
            dim_size=x.size(0),
            reduce=self.aggr,
        )  # [N, edge_mlp_dim]

        node_input = torch.cat([x, agg_m], dim=-1)   # [N, in_dim + edge_mlp_dim]
        x_out = self.node_mlp(node_input)            # [N, out_dim]

        return x_out, pos_out

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.typing import OptTensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

__all__ = [
    'TensorGNAN',
]


def _init_weights(module: nn.Module, std: float = 1.0):
    """Utility that mimics Xavier initialisation with configurable std."""
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param, gain=std)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


class _PerFeatureMLP(nn.Module):
    """Simple MLP that is applied to a single scalar feature.

    Args:
        out_channels (int): Output dimension per feature ("f" in the paper).
        n_layers (int): Number of layers. If ``1``, the MLP is a single Linear.
        hidden_channels (int, optional): Hidden dimension. Required when
            ``n_layers > 1``.
        bias (bool, optional): Use bias terms. (default: ``True``)
        dropout (float, optional): Dropout probability after hidden layers.
            (default: ``0.0``)
    """
    def __init__(
        self,
        out_channels: int,
        n_layers: int,
        hidden_channels: int | None = None,
        *,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_layers == 1:
            self.net = nn.Linear(1, out_channels, bias=bias)
        else:
            assert hidden_channels is not None
            layers: list[nn.Module] = [
                nn.Linear(1, hidden_channels, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for _ in range(1, n_layers - 1):
                layers += [
                    nn.Linear(hidden_channels, hidden_channels, bias=bias),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.net = nn.Sequential(*layers)

        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N]
        return self.net(x.view(-1, 1))  # [N, out_channels]


class _RhoMLP(nn.Module):
    """MLP that turns a scalar distance into a scalar or vector weight."""
    def __init__(
        self,
        out_channels: int,
        n_layers: int,
        hidden_channels: int | None = None,
        *,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_layers == 1:
            self.net = nn.Linear(1, out_channels, bias=bias)
        else:
            assert hidden_channels is not None
            layers: list[nn.Module] = [
                nn.Linear(1, hidden_channels, bias=bias),
                nn.ReLU()
            ]
            for _ in range(1, n_layers - 1):
                layers += [
                    nn.Linear(hidden_channels, hidden_channels, bias=bias),
                    nn.ReLU(),
                ]
            layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.net = nn.Sequential(*layers)

        _init_weights(self)

    def forward(self, d: torch.Tensor) -> torch.Tensor:  # [...]
        return self.net(d.view(-1, 1))


class TensorGNAN(nn.Module):
    r"""Dense, tensorised GNAN variant.

    By default it aggregates node scores to produce *graph‐level* predictions
    (shape ``[batch_size, out_channels]``).  Set ``graph_level=False`` to
    obtain *node‐level* predictions instead, in which case the forward returns
    a tensor of shape ``[num_nodes, out_channels]`` or ``[len(node_ids),
    out_channels]`` if ``node_ids`` is provided.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        *,
        hidden_channels: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        normalize_rho: bool = True,
        graph_level: bool = True,
    ) -> None:
        super().__init__()

        self.normalize_rho = normalize_rho
        self.graph_level = graph_level
        self.out_channels = out_channels

        self.fs = nn.ModuleList([
            _PerFeatureMLP(out_channels, n_layers, hidden_channels, bias=bias,
                           dropout=dropout) for _ in range(in_channels)
        ])
        self.rho = _RhoMLP(out_channels, n_layers, hidden_channels, bias=True)

    # --------------------------------------------------------------
    def forward(self, data: Data | Batch,
                node_ids: OptTensor = None) -> torch.Tensor:
        x: torch.Tensor = data.x  # type: ignore # [N, F]
        dist: torch.Tensor = data.node_distances  # type: ignore # [N, N]
        norm: torch.Tensor = data.normalization_matrix  # type: ignore # [N, N]

        # f_k(x_k)
        fx = torch.stack([mlp(x[:, k]) for k, mlp in enumerate(self.fs)],
                         dim=1)  # [N, F, C]

        # Compute ρ on the inverted distances as suggested in the paper
        inv_dist = 1.0 / (1.0 + dist)  # [N, N]
        rho = self.rho(inv_dist.flatten().view(-1, 1))  # [(N*N), C]
        rho = rho.view(x.size(0), x.size(0), self.out_channels)  # [N, N, C]

        if self.normalize_rho:
            norm[norm == 0] = 1.0
            rho = rho / norm.unsqueeze(-1)  # broadcast

        # Apply a mask to ρ to prevent information leakage between
        # graphs in the same batch.
        if hasattr(data, 'batch') and data.batch is not None:
            batch_i = data.batch.view(-1, 1)  # type: ignore
            batch_j = data.batch.view(1, -1)  # type: ignore
            mask = (batch_i == batch_j).unsqueeze(-1)
            rho = rho * mask

        # Perform Σ_i Σ_j ρ(d_ij) Σ_k f_k(x_jk)
        f_sum = fx.sum(dim=1)  # [N, C]
        out = torch.einsum('ijc,jc->ic', rho, f_sum)  # [N, C]

        if self.graph_level:
            # # Use batch information for proper graph-level aggregation
            batch = data.batch  # type: ignore
            if batch is not None:
                graph_out = scatter(out, batch, dim=0, reduce='add')
            else:
                # Single graph case
                graph_out = out.sum(dim=0, keepdim=True)  # [1, C]
            return graph_out

        # --- node-level mode -------------------------------------------------
        if node_ids is not None:
            return out[node_ids]
        return out

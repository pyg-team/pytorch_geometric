from __future__ import annotations

import torch
from torch import nn

from torch_geometric.data import Batch, Data
from torch_geometric.typing import OptTensor
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


class _MultiFeatureMLP(nn.Module):
    """MLP that processes multiple features together.

    Args:
        in_channels (int): Number of input features to process together.
        out_channels (int): Output dimension per feature group.
        n_layers (int): Number of layers. If ``1``, the MLP is a single Linear.
        hidden_channels (int, optional): Hidden dimension. Required when
            ``n_layers > 1``.
        bias (bool, optional): Use bias terms. (default: ``True``)
        dropout (float, optional): Dropout probability after hidden layers.
            (default: ``0.0``)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        hidden_channels: int | None = None,
        *,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_layers == 1:
            self.net = nn.Linear(in_channels, out_channels, bias=bias)
        else:
            assert hidden_channels is not None
            layers: list[nn.Module] = [
                nn.Linear(in_channels, hidden_channels, bias=bias),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, in_channels]
        return self.net(x)  # [N, out_channels]


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
    
    Args:
        in_channels (int): Number of input node features.
        out_channels (int): Output dimension.
        n_layers (int): Number of layers in the MLPs.
        hidden_channels (int, optional): Hidden dimension in the MLPs.
        bias (bool, optional): Use bias terms. (default: ``True``)
        dropout (float, optional): Dropout probability. (default: ``0.0``)
        normalize_rho (bool, optional): Whether to normalize rho weights.
            (default: ``True``)
        graph_level (bool, optional): Whether to produce graph-level predictions.
            (default: ``True``)
        feature_groups (List[List[int]], optional): Groups of feature indices to
            process together. Each group will be processed by a single MLP that
            takes multiple features as input. If None, each feature is processed
            by its own MLP (default behavior). (default: ``None``)
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
        feature_groups: list[list[int]] | None = None,
    ) -> None:
        super().__init__()

        self.normalize_rho = normalize_rho
        self.graph_level = graph_level
        self.out_channels = out_channels
        self.in_channels = in_channels

        # Set up feature groups - default is each feature in its own group
        if feature_groups is None:
            self.feature_groups = [[i] for i in range(in_channels)]
        else:
            self.feature_groups = feature_groups
            # Validate feature groups
            all_features = set()
            for group in feature_groups:
                if not group:
                    raise ValueError("Feature groups cannot be empty")
                for feat_idx in group:
                    if feat_idx < 0 or feat_idx >= in_channels:
                        raise ValueError(f"Feature index {feat_idx} out of range [0, {in_channels})")
                    if feat_idx in all_features:
                        raise ValueError(f"Feature index {feat_idx} appears in multiple groups")
                    all_features.add(feat_idx)
            
            if len(all_features) != in_channels:
                missing = set(range(in_channels)) - all_features
                raise ValueError(f"Missing feature indices in groups: {missing}")

        # Create MLPs for each feature group
        self.fs = nn.ModuleList()
        for group in self.feature_groups:
            group_size = len(group)
            if group_size == 1:
                # Single feature - use original MLP
                mlp = _PerFeatureMLP(out_channels, n_layers, hidden_channels, 
                                   bias=bias, dropout=dropout)
            else:
                # Multiple features - use new multi-feature MLP
                mlp = _MultiFeatureMLP(group_size, out_channels, n_layers, 
                                     hidden_channels, bias=bias, dropout=dropout)
            self.fs.append(mlp)

        self.rho = _RhoMLP(out_channels, n_layers, hidden_channels, bias=True)

    def _process_feature_groups(self, x: torch.Tensor):
        """Process features according to groups and return fx and f_sum."""
        fx_list: list[torch.Tensor] = []
        for group, mlp in zip(self.feature_groups, self.fs):
            if len(group) == 1:
                feat_tensor = x[:, group[0]]  # [N]
            else:
                feat_tensor = x[:, group]      # [N, |group|]
            fx_list.append(mlp(feat_tensor))   # [N, C]
        fx = torch.stack(fx_list, dim=1)        # [N, num_groups, C]
        f_sum = fx.sum(dim=1)                   # [N, C]
        return fx, f_sum

    def _compute_rho(self, dist: torch.Tensor, norm: torch.Tensor, data) -> torch.Tensor:
        """Compute rho tensor, including normalization and masking."""
        x = data.x  # type: ignore
        inv_dist = 1.0 / (1.0 + dist)  # [N, N]
        rho = self.rho(inv_dist.flatten().view(-1, 1))  # [(N*N), C]
        rho = rho.view(x.size(0), x.size(0), self.out_channels)  # [N, N, C]
        if self.normalize_rho:
            norm_safe = norm.clone()
            norm_safe[norm_safe == 0] = 1.0
            rho = rho / norm_safe.unsqueeze(-1)  # broadcast division
        if hasattr(data, 'batch') and data.batch is not None:
            batch_i = data.batch.view(-1, 1)  # type: ignore
            batch_j = data.batch.view(1, -1)  # type: ignore
            mask = (batch_i == batch_j).unsqueeze(-1)
            rho = rho * mask
        return rho

    def forward(self, data: Data | Batch,
                node_ids: OptTensor = None) -> torch.Tensor:
        x: torch.Tensor = data.x  # type: ignore # [N, F]
        dist: torch.Tensor = data.node_distances  # type: ignore # [N, N]
        norm: torch.Tensor = data.normalization_matrix  # type: ignore # [N, N]

        _, f_sum = self._process_feature_groups(x)
        rho = self._compute_rho(dist, norm, data)

        # Perform Σ_i Σ_j ρ(d_ij) Σ_k f_k(x_jk)
        out = torch.einsum('ijc,jc->ic', rho, f_sum)  # [N, C]

        if self.graph_level:
            batch = data.batch  # type: ignore
            if batch is not None:
                graph_out = scatter(out, batch, dim=0, reduce='add')
            else:
                graph_out = out.sum(dim=0, keepdim=True)  # [1, C]
            return graph_out

        if node_ids is not None:
            return out[node_ids]
        return out

    def node_importance(self, data: Data | Batch) -> torch.Tensor:
        """Returns the  contribution of every node to the
        graph‐level prediction. UsingEq. (3) in the paper.
        """
        x: torch.Tensor = data.x  # type: ignore  # [N, F]
        dist: torch.Tensor = data.node_distances  # type: ignore  # [N, N]
        norm: torch.Tensor = data.normalization_matrix  # type: ignore  # [N, N]

        _, f_sum = self._process_feature_groups(x)
        rho = self._compute_rho(dist, norm, data)

        # Aggregate over *receiver* nodes i to obtain \sum_i rho(d_{ij}).
        rho_sum_over_i = rho.sum(dim=0)          # [N, C]

        # Node contribution s_j = (sum_k f_k(x_jk)) * (sum_i rho(d_{ij})).
        node_contrib = f_sum * rho_sum_over_i    # [N, C]

        return node_contrib

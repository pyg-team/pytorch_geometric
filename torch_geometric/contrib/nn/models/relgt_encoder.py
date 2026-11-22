# Adapted from snap-stanford/relgt, encoders.py and model.py, commit
# 19e423ca3e7cac761130aba790857f2dc3a46ef7 (MIT).
# https://github.com/snap-stanford/relgt
# See relgt.LICENSE in this directory for the original license notice.

from typing import Optional

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleDict,
    ModuleList,
    Parameter,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINConv, PositionalEncoding


class _RelGTStructureEncoder(Module):
    def __init__(self, channels: int, num_layers: int):
        super().__init__()
        hidden_channels = channels // 4
        self.input_proj = Linear(1, hidden_channels)
        self.conv = ModuleList([
            GINConv(
                Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                ), train_eps=True) for _ in range(num_layers)
        ])
        self.bns = ModuleList(
            [BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.final_transform = Linear(hidden_channels, channels)

    def reset_parameters(self) -> None:
        for projection in [self.input_proj, self.final_transform]:
            torch.nn.init.xavier_uniform_(projection.weight)
            torch.nn.init.zeros_(projection.bias)
        for conv, norm in zip(self.conv, self.bns):
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(self, noise: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_proj(noise)
        for conv, norm in zip(self.conv, self.bns):
            x = x + norm(conv(x, edge_index)).relu()
        return self.final_transform(x)


class RelGTTokenEncoder(Module):
    r"""Five-part token encoder from the `"Relational Graph Transformer"
    <https://arxiv.org/abs/2505.10960>`_ paper.

    Combines differentiable row embeddings with table, hop, age, and graph
    structure encodings. Each context has a seed at position zero. The
    implementation follows the reference's normalized concatenation and
    nonlinear mixing, including its positional encoding of age in days.

    Structural noise is supplied by the caller in both training and
    evaluation. Resample it during training and fix it for evaluation.
    Graph nodes identify token occurrences, so repeated entities retain
    distinct positions. No temporal sampling or feature encoding occurs here.

    Args:
        channels (int): Token width, a positive multiple of four.
        num_node_types (int): Number of known table types. The additional
            index :obj:`num_node_types` denotes an unknown or masked type.
        max_hop (int, optional): Maximum hop code. Code three denotes a
            fallback row in the reference pipeline. (default: :obj:`3`)
        pe_layers (int, optional): Number of residual structural GIN layers.
            (default: :obj:`4`)
        ablation (str, optional): Remove :obj:`"time"` or
            :obj:`"structure"` before mixing. (default: :obj:`None`)
    """
    def __init__(self, channels: int, num_node_types: int, max_hop: int = 3, *,
                 pe_layers: int = 4, ablation: Optional[str] = None):
        super().__init__()
        for name, value in [('channels', channels),
                            ('num_node_types', num_node_types),
                            ('pe_layers', pe_layers)]:
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"'{name}' must be a positive integer")
        if channels % 4 != 0:
            raise ValueError("'channels' must be divisible by four")
        if (not isinstance(max_hop, int) or isinstance(max_hop, bool)
                or max_hop < 0):
            raise ValueError("'max_hop' must be a nonnegative integer")
        if ablation not in [None, 'time', 'structure']:
            raise ValueError("'ablation' must be None, 'time', or 'structure'")

        self.channels = channels
        self.num_node_types = num_node_types
        self.max_hop = max_hop
        self.pe_layers = pe_layers
        self.ablation = ablation
        self.type_embedding = Embedding(num_node_types + 1, channels)
        self.hop_embedding = Embedding(max_hop + 2, channels)
        self.time_encoding = None
        self.time_projection = None
        self.register_parameter('time_mask', None)
        if ablation != 'time':
            self.time_encoding = PositionalEncoding(channels)
            self.time_projection = Linear(channels, channels)
            self.time_mask = Parameter(torch.empty(channels))
        self.pe_encoder = None
        if ablation != 'structure':
            self.pe_encoder = _RelGTStructureEncoder(channels, pe_layers)
        names = ['node_type', 'hop', 'time', 'features', 'structure']
        self.norms = ModuleDict(
            {name: LayerNorm(channels)
             for name in names if name != ablation})
        self.in_mixture = Sequential(
            Linear(len(self.norms) * channels, 2 * channels),
            ReLU(),
            Linear(2 * channels, channels),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets parameters, GIN epsilons, and normalization statistics."""
        self.type_embedding.reset_parameters()
        self.hop_embedding.reset_parameters()
        if self.time_projection is not None:
            self.time_encoding.reset_parameters()
            self.time_projection.reset_parameters()
            torch.nn.init.normal_(self.time_mask, std=0.02)
        if self.pe_encoder is not None:
            self.pe_encoder.reset_parameters()
        for norm in self.norms.values():
            norm.reset_parameters()
        for layer in self.in_mixture:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _validate(self, x: Tensor, node_type: Tensor, hop: Tensor, age: Tensor,
                  edge_index: Tensor, pe_noise: Optional[Tensor]) -> None:
        if x.dim() != 3 or x.size(-1) != self.channels:
            raise ValueError("'x' must have shape [B, K, channels]")
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("'x' must contain at least one context/token")
        if not x.is_floating_point():
            raise TypeError("'x' must be floating point")
        if not torch.isfinite(x).all():
            raise ValueError("'x' must be finite")
        for name, tensor in [('node_type', node_type), ('hop', hop),
                             ('age', age)]:
            if tensor.shape != x.shape[:2]:
                raise ValueError(f"'{name}' must have shape [B, K]")
            if tensor.device != x.device:
                raise ValueError(f"'{name}' must be on the same device as 'x'")
        if node_type.dtype != torch.long or hop.dtype != torch.long:
            raise TypeError("'node_type' and 'hop' must have dtype torch.long")
        if (node_type < 0).any() or (node_type > self.num_node_types).any():
            raise ValueError("'node_type' is outside the supported range")
        if (hop < -1).any() or (hop > self.max_hop).any():
            raise ValueError("'hop' is outside the supported range")
        if not age.is_floating_point():
            raise TypeError("'age' must be floating point")
        if (not torch.isfinite(age).all() or ((age < 0) & (age != -1)).any()):
            raise ValueError("'age' must be finite, nonnegative, or -1")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("'edge_index' must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise TypeError("'edge_index' must have dtype torch.long")
        if edge_index.device != x.device:
            raise ValueError("'edge_index' must be on the same device as 'x'")
        if edge_index.numel() > 0:
            if (edge_index < 0).any() or (edge_index
                                          >= x.size(0) * x.size(1)).any():
                raise ValueError(
                    "'edge_index' contains an invalid token index")
            if (edge_index[0] // x.size(1)
                    != edge_index[1] // x.size(1)).any():
                raise ValueError("'edge_index' must not connect contexts")
        if self.pe_encoder is not None:
            if pe_noise is None:
                raise ValueError("'pe_noise' is required when structure is "
                                 "active")
            if pe_noise.shape != (*x.shape[:2], 1):
                raise ValueError("'pe_noise' must have shape [B, K, 1]")
            if pe_noise.device != x.device or pe_noise.dtype != x.dtype:
                raise ValueError("'pe_noise' must match the device/dtype of "
                                 "'x'")
            if not torch.isfinite(pe_noise).all():
                raise ValueError("'pe_noise' must be finite")
            if self.training and x.size(0) * x.size(1) < 2:
                raise ValueError("Training requires at least two token "
                                 "occurrences")

    def forward(
        self,
        x: Tensor,
        node_type: Tensor,
        hop: Tensor,
        age: Tensor,
        edge_index: Tensor,
        pe_noise: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Constructs tokens with shape :obj:`[B, K, channels]`.

        Args:
            x (torch.Tensor): Row embeddings, :obj:`[B, K, channels]`.
            node_type (torch.Tensor): Table IDs, :obj:`[B, K]`.
            hop (torch.Tensor): Hop codes, :obj:`[B, K]`; :obj:`-1` is unknown.
            age (torch.Tensor): Elapsed days, :obj:`[B, K]`;
                :obj:`-1` is unknown.
            edge_index (torch.Tensor): Edges over flattened token occurrences,
                :obj:`[2, E]`. Edges cannot connect different contexts.
            pe_noise (torch.Tensor, optional): Explicit structural noise,
                :obj:`[B, K, 1]`. Required unless structure is ablated.
        """
        self._validate(x, node_type, hop, age, edge_index, pe_noise)
        components = [
            self.norms['node_type'](self.type_embedding(node_type)),
            self.norms['hop'](self.hop_embedding(hop + 1)),
        ]
        if self.time_projection is not None:
            encoded = self.time_encoding(age.clamp_min(0).reshape(-1))
            time = self.time_projection(encoded).reshape_as(x)
            time = torch.where((age == -1).unsqueeze(-1), self.time_mask, time)
            components.append(self.norms['time'](time))
        components.append(self.norms['features'](x))
        if self.pe_encoder is not None:
            structure = self.pe_encoder(pe_noise.reshape(-1, 1), edge_index)
            components.append(self.norms['structure'](structure.reshape_as(x)))
        return self.in_mixture(torch.cat(components, dim=-1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(channels={self.channels}, '
                f'num_node_types={self.num_node_types}, '
                f'max_hop={self.max_hop}, '
                f'pe_layers={self.pe_layers}, ablation={self.ablation!r})')

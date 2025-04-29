from typing import Callable, Literal, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import ModuleList

from torch_geometric.contrib.nn.bias.base import BaseBiasProvider
from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.utils.mask_utils import build_key_padding
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class GraphTransformer(torch.nn.Module):
    r"""The graph transformer model from the "Transformer for Graphs:
    An Overview from Architecture Perspective"
    <https://arxiv.org/pdf/2202.08455>_ paper.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        num_class: int = 2,
        use_super_node: bool = False,
        node_feature_encoder=nn.Identity(),
        num_encoder_layers: int = 0,
        degree_encoder: Optional[Callable[[Data], torch.Tensor]] = None,
        attn_bias_providers: Sequence[BaseBiasProvider] = (),
        gnn_block: Optional[Callable[[Data, torch.Tensor],
                                     torch.Tensor]] = None,
        gnn_position: Literal['pre', 'post', 'parallel'] = 'pre'
    ) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.use_super_node = use_super_node
        if self.use_super_node:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.node_feature_encoder = node_feature_encoder
        self.degree_encoder = degree_encoder
        encoder_layer = GraphTransformerEncoderLayer(hidden_dim)
        self.encoder = (
            GraphTransformerEncoder(encoder_layer, num_encoder_layers)
            if num_encoder_layers > 0 else encoder_layer
        )
        self.is_encoder_stack = num_encoder_layers > 0
        self.attn_bias_providers = ModuleList(attn_bias_providers)
        self.gnn_block = gnn_block
        self.gnn_position = gnn_position

    @torch.jit.ignore
    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.use_super_node:
            graph_sizes = torch.bincount(batch)
            first_idx = torch.cumsum(graph_sizes, 0) - graph_sizes
            return x[first_idx]
        else:
            return global_mean_pool(x, batch)

    @torch.jit.ignore
    def _add_cls_token(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Prepends the learnable class token to every graph's node embeddings.

        Args:
            x (torch.Tensor): The input features.
            batch (torch.Tensor): The batch vector.

        Returns:
            torch.Tensor: A stacked tensor of shape
            (num_graphs, num_nodes_i+1, hidden_dim) where
            the first token of every graph corresponds to the cls token.
        """
        num_graphs = batch.max().item() + 1
        x_list = []
        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask]
            x_i = torch.cat([self.cls_token.expand(1, -1), x_i], dim=0)
            x_list.append(x_i)
        return torch.stack(x_list, dim=0)

    @torch.jit.ignore
    def _prepend_cls_token_flat(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_with_cls_flat, new_batch).

        Args:
            x (torch.Tensor): Node feature tensor (N, C)
            batch (torch.Tensor): Batch assignment (N,)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Features with prepended
            CLS tokens and updated batch vector
        """
        num_graphs = int(batch.max()) + 1
        x_list = []
        b_list = []

        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask]
            x_with_cls = torch.cat([self.cls_token, x_i], dim=0)
            x_list.append(x_with_cls)
            b_list.append(torch.full((len(x_i) + 1, ), i, device=x.device))

        return torch.cat(x_list, dim=0), torch.cat(b_list, dim=0)

    @torch.jit.ignore
    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes node features using the node feature encoder.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The encoded node features.
        """
        if self.node_feature_encoder is not None:
            x = self.node_feature_encoder(x)
        return x

    @torch.jit.ignore
    def _apply_extra_encoders(
        self, data: Data, x: torch.Tensor
    ) -> torch.Tensor:
        """Apply additional encoders (e.g. degree encoder) to node features.

        Args:
            data (Data): The input graph data.
            x (torch.Tensor): Current node features.

        Returns:
            torch.Tensor: Node features with extra encodings applied.
        """
        if self.degree_encoder is not None:
            deg_feat = self.degree_encoder(data).to(x.device)
            x = x + deg_feat
        return x

    def forward(self, data):
        r"""Applies the graph transformer model to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            torch.Tensor: The output of the model.
        """
        x = data.x
        x = self._encode_nodes(x)
        x = self._apply_extra_encoders(data, x)

        if self.gnn_block is not None and self.gnn_position == 'pre':
            x = self.gnn_block(data, x)
        if self.use_super_node:
            x, batch_vec = self._prepend_cls_token_flat(x, data.batch)
        else:
            batch_vec = data.batch

        struct_mask = self._collect_attn_bias(data)

        if self.gnn_block and self.gnn_position == 'parallel':
            x_in = x

        if self.is_encoder_stack:
            x = self.encoder(x, batch_vec, struct_mask)
        else:
            num_heads = getattr(self.encoder, "num_heads", None)
            if num_heads is not None:
                key_pad = build_key_padding(batch_vec, num_heads=num_heads)
                x = self.encoder(x, batch_vec, struct_mask, key_pad)
            else:
                x = self.encoder(x, batch_vec, struct_mask)

        if self.gnn_block is not None and self.gnn_position == 'post':
            x = self.gnn_block(data, x)

        if self.gnn_block and self.gnn_position == 'parallel':
            x = x + self.gnn_block(data, x_in)

        x = self._readout(x, batch_vec)
        logits = self.classifier(x)
        return {
            "logits": logits,
        }

    @torch.jit.ignore
    def _collect_attn_bias(self, data: Data) -> Optional[torch.Tensor]:
        """Combine legacy and all provider masks into one.

        Returns:
            FloatTensor of shape (B, H, L, L) or None if no mask was provided.
        """
        masks = []
        if getattr(data, "bias", None) is not None:
            legacy = data.bias
            # cast bool→float, leave floats as-is
            masks.append(
                legacy.to(torch.float32)
                if not torch.is_floating_point(legacy) else legacy
            )

        for prov in self.attn_bias_providers:
            m = prov(data)
            if m is not None:
                masks.append(m.to(torch.float32))

        if not masks:
            return None
        return torch.stack(
            masks, dim=0
        ).sum(dim=0).to(data.x.dtype).to(data.x.device)

    def __repr__(self):
        n_layers = len(self.encoder) if self.is_encoder_stack else 0
        providers = [
            prov.__class__.__name__ for prov in self.attn_bias_providers
        ]
        if self.gnn_block is not None:
            gnn_name = self.gnn_block.__name__
        else:
            gnn_name = None

        return (
            "GraphTransformer("
            f"hidden_dim={self.classifier.in_features}, "
            f"num_class={self.classifier.out_features}, "
            f"use_super_node={self.use_super_node}, "
            f"num_encoder_layers={n_layers}, "
            f"bias_providers={providers}, "
            f"gnn_hook={gnn_name}@'{self.gnn_position}'"
            ")"
        )

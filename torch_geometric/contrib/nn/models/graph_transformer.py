from typing import Callable, Optional

import torch
import torch.nn as nn

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

        if self.use_super_node:
            x, batch_vec = self._prepend_cls_token_flat(x, data.batch)
        else:
            batch_vec = data.batch
        attn_mask = getattr(data, 'attn_mask', None)
        if self.is_encoder_stack:
            x = self.encoder(x, batch_vec, attn_mask)
        else:
            num_heads = getattr(self.encoder, "num_heads", None)
            if num_heads is not None:
                key_pad = build_key_padding(batch_vec, num_heads=num_heads)
                x = self.encoder(x, batch_vec, attn_mask, key_pad)
            else:
                x = self.encoder(x, batch_vec, attn_mask)

        x = self._readout(x, batch_vec)
        logits = self.classifier(x)
        return {
            "logits": logits,
        }

    def __repr__(self):
        return "GraphTransformer()"

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.contrib.nn.layers.feedforward import (
    PositionwiseFeedForward,
)
from torch_geometric.contrib.utils.mask_utils import (
    build_key_padding,
    merge_masks,
)


class GraphTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_hidden_dim: Optional[int] = None,
        activation='gelu'
    ):
        """Initialize a GraphTransformerEncoderLayer.

        Args:
            hidden_dim (int): Dimension of the input features
            num_heads (int, optional): Number of attention heads.
                Defaults to 4.
            dropout (float, optional): Dropout probability.
                Defaults to 0.1.
            ffn_hidden_dim (int, optional): Hidden dimension of FFN. If None,
                                          defaults to 4 * hidden_dim.
            activation (str, optional): Activation function.
                Defaults to 'gelu'.

        Note:
            self.self_attn is initialized as nn.MultiheadAttention.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim or 4 * hidden_dim
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = PositionwiseFeedForward(
            hidden_dim, self.ffn_hidden_dim, dropout, activation
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )

    def _pad_to_dense(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert flat node features to dense padded batch.

        Args:
            x (torch.Tensor): Node features (total_nodes, hidden_dim)
            batch (torch.Tensor): Batch indices (total_nodes,)

        Returns:
            tuple:
                dense(torch.Tensor): Padded tensor of shape
                    (batch_size, max_nodes, hidden_dim)
                lengths(torch.Tensor): Nodes per graph (batch_size,)
        """
        batch_size = int(batch.max() + 1)
        lengths = torch.bincount(batch)
        max_nodes = lengths.max()
        dense = torch.zeros(
            batch_size, max_nodes, x.size(-1), device=x.device, dtype=x.dtype
        )
        node_indices = torch.arange(max_nodes, device=x.device)
        batch_idx, node_idx = torch.where(
            node_indices.unsqueeze(0) < lengths.unsqueeze(1)
        )
        offsets = torch.nn.functional.pad(
            torch.cumsum(lengths[:-1], 0), (1, 0)
        )
        src_idx = offsets[batch_idx] + node_idx
        dense[batch_idx, node_idx] = x[src_idx]

        return dense, lengths

    def _unpad(
        self, dense: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Convert dense padded batch back to flat tensor.

        Args:
            dense (torch.Tensor): Padded tensor of shape
            (batch_size, max_nodes, hidden_dim)
            lengths (torch.Tensor): Nodes per graph

        Returns:
            torch.Tensor: Flat tensor (total_nodes, hidden_dim)
        """
        total_nodes = lengths.sum().item()
        out = torch.zeros(
            total_nodes,
            dense.size(-1),
            device=dense.device,
            dtype=dense.dtype
        )

        batch_idx, node_idx = torch.where(
            torch.arange(lengths.max(), device=dense.device).unsqueeze(0) <
            lengths.unsqueeze(1)
        )
        offsets = torch.nn.functional.pad(
            torch.cumsum(lengths[:-1], 0), (1, 0)
        )
        tgt_idx = offsets[batch_idx] + node_idx
        out[tgt_idx] = dense[batch_idx, node_idx]

        return out

    def forward(
        self, x: Tensor, batch: Tensor, attn_mask: Optional[Tensor],
        key_pad: Tensor
    ):
        """Forward pass through transformer encoder layer.

        Args:
            x (Tensor): Node features (total_nodes, hidden_dim)
            batch (Tensor): Batch indices (total_nodes,)
            attn_mask (Optional[Tensor]): Attention mask
            key_pad (Tensor): Pre-computed key padding mask

        Returns:
            Tensor: Transformed features (total_nodes, hidden_dim)
        """
        assert x.dim() == 2, \
            f"Expected 2D input tensor (N,C), got shape {x.shape}"
        x_batch, lengths = self._pad_to_dense(x, batch)
        additive = merge_masks(
            key_pad=key_pad,
            attn=attn_mask,
            num_heads=self.num_heads,
            dtype=x.dtype
        )

        x = self.norm1(x_batch)
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=additive, need_weights=False
        )
        x = x_batch + self.dropout_layer(attn_out)
        x_flat = self._unpad(x, lengths)
        src2 = x_flat
        x = self.norm2(x_flat)
        ffn_out = self.ffn(x)
        x = src2 + self.dropout_layer(ffn_out)

        return x


class GraphTransformerEncoder(nn.Module):
    """A stack of N encoder layers."""

    def __init__(self, encoder_layer, num_layers):
        """Initialize the encoder with a specified number of identical layers.

        Args:
            encoder_layer (GraphTransformerEncoderLayer): The encoder layer
            to be stacked
            num_layers (int): The number of encoder layers to stack
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                type(encoder_layer)(
                    hidden_dim=encoder_layer.hidden_dim,
                    num_heads=encoder_layer.num_heads,
                    dropout=encoder_layer.dropout_layer.p,
                    ffn_hidden_dim=encoder_layer.ffn_hidden_dim,
                ) for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.num_heads = encoder_layer.num_heads

    def forward(self, x, batch, attn_mask=None):
        """Apply all encoder layers in sequence.

        If all layers share the same num_heads, build the pad once;
        if any layer has a different head‐count, rebuild it before that layer.

        Args:
            x (torch.Tensor): Node feature tensor
            batch (torch.Tensor): Batch vector
            attn_mask (torch.Tensor, optional): Attention mask

        Returns:
            torch.Tensor: Output after passing through all encoder layers.
        """
        current_h = None
        key_pad = None
        for layer in self.layers:
            if key_pad is None or layer.num_heads != current_h:
                key_pad = build_key_padding(batch, num_heads=layer.num_heads)
                current_h = layer.num_heads
            x = layer(x, batch, attn_mask, key_pad)
        return x

    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx):
        """Get encoder layer at index idx.

        Args:
            idx (int): Layer index

        Returns:
            nn.Module: The encoder layer at index idx
        """
        return self.layers[idx]

    def __setitem__(self, idx, layer):
        """Set encoder layer at index idx.

        Args:
            idx (int): Layer index
            layer (nn.Module): New layer to set at index idx
        """
        self.layers[idx] = layer

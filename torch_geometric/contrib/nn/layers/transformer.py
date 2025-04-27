from typing import Optional

import torch
import torch.nn as nn

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
        self.norm1 = nn.LayerNorm(hidden_dim)  # Normalization before attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # Normalization before FFN
        self.activation = activation
        self.ffn = PositionwiseFeedForward(
            hidden_dim, self.ffn_hidden_dim, dropout, activation
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        self.key_padding = None

    def forward(self, x, batch=None, attn_mask=None):
        """Forward pass through the transformer encoder layer.

        Args:
            x (torch.Tensor): Node feature tensor (total_nodes, hidden_dim)
            batch (torch.Tensor, optional): Batch vector (total_nodes,)
            attn_mask (torch.Tensor, optional): Attention mask tensor

        Returns:
            torch.Tensor: Transformed node features (total_nodes, hidden_dim)
        """
        assert x.dim() == 2, f"Expected 2D input tensor (N,C),\
            got shape {x.shape}"

        src = x
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        batch_size = int(batch.max() + 1)
        max_nodes = torch.bincount(batch).max()

        x_batch = torch.zeros(
            batch_size,
            max_nodes,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype
        )
        for i in range(batch_size):
            mask = batch == i
            num_nodes = mask.sum()
            x_batch[i, :num_nodes] = x[mask]

        additive = merge_masks(
            key_pad=build_key_padding(batch, num_heads=self.num_heads),
            attn=attn_mask,
            num_heads=self.num_heads,
            dtype=x.dtype
        )

        # Apply attention
        x = self.norm1(x_batch)
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=additive, need_weights=False
        )
        x = x_batch + self.dropout_layer(attn_out)

        # Flatten back to (total_nodes, hidden_dim)
        x_flat = torch.zeros_like(src)
        for i in range(batch_size):
            mask = batch == i
            num_nodes = mask.sum()
            x_flat[mask] = x[i, :num_nodes]

        # Apply FFN
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
                    activation=encoder_layer.activation
                ) for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(self, x, batch=None, attn_mask=None):
        """Apply all encoder layers in sequence.

        Args:
            x (torch.Tensor): Node feature tensor
            batch (torch.Tensor, optional): Batch vector
            attn_mask (torch.Tensor, optional): Attention mask

        Returns:
            torch.Tensor: Output after passing through all encoder layers
        """
        for layer in self.layers:
            x = layer(x, batch, attn_mask)
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

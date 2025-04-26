import copy
from typing import Optional

import torch
import torch.nn as nn

from torch_geometric.contrib.nn.layers.feedforward import (
    PositionwiseFeedForward,
)


class GraphTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_mask: Optional[torch.Tensor] = None,
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
            attn_mask (torch.Tensor, optional): Attention mask.
                Defaults to None.
            ffn_hidden_dim (int, optional): Hidden dimension of FFN. If None,
                                          defaults to 4 * hidden_dim.
            activation (str, optional): Activation function.
                Defaults to 'gelu'.
            attn_mask (torch.Tensor, optional): Attention mask.
                Defaults to None.

        Note:
            self.self_attn is initialized as nn.Identity() and should be
            replaced with an actual attention mechanism implementation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim or 4 * hidden_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)  # Normalization before attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # Normalization before FFN
        self.activation = activation
        self.ffn = PositionwiseFeedForward(
            hidden_dim, self.ffn_hidden_dim, dropout, activation
        )
        self.self_attn = nn.Identity()
        self.attn_mask = attn_mask
        self._static_attn_mask = None  # will be used when add real attention

    def forward(self, x, batch=None, attn_mask=None):
        """Forward pass through the transformer encoder layer.

        Args:
            x (torch.Tensor): Node feature tensor
            batch (torch.Tensor, optional): Batch vector
            attn_mask (torch.Tensor, optional): Attention mask tensor

        Returns:
            torch.Tensor: Transformed node features
        """
        src = x
        if attn_mask is None:
            attn_mask = self._static_attn_mask
        x = self.norm1(x)
        # TODO: pass attn_mask into self.self_attn when it becomes real
        attn_out = self.self_attn(x)
        x = src + self.dropout_layer(attn_out)
        src2 = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = src2 + self.dropout_layer(ffn_out)
        return x

    @staticmethod
    @torch.jit.ignore
    def _build_key_padding_mask(batch: torch.Tensor,
                                max_len: int) -> Optional[torch.Tensor]:
        """Build key padding mask from batch vector.

        Args:
            batch (torch.Tensor): Batch assignment vector
            max_len (int): Maximum sequence length

        Returns:
            Optional[torch.Tensor]: Key padding mask, currently returns None
        """
        return None


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
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, x, batch=None):
        """Apply all encoder layers in sequence.

        Args:
            x (torch.Tensor): Node feature tensor
            batch (torch.Tensor, optional): Batch vector

        Returns:
            torch.Tensor: Output after passing through all encoder layers
        """
        for layer in self.layers:
            x = layer(x, batch)
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

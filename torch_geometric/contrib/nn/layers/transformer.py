from typing import Optional

import torch.nn as nn
from torch import Tensor

# Use alias imports to avoid yapf/isort conflicts with long module names
import torch_geometric.contrib.nn.layers.feedforward as _feedforward
import torch_geometric.contrib.utils.mask_utils as _mask_utils
from torch_geometric.nn.inits import reset

# Extract classes/functions from alias imports
PositionwiseFeedForward = _feedforward.PositionwiseFeedForward
build_key_padding = _mask_utils.build_key_padding
merge_masks = _mask_utils.merge_masks


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, ffn_hidden_dim: Optional[int] = None,
                 activation='gelu'):
        """Initialize a GraphTransformerEncoderLayer.

        This layer implements a transformer encoder layer with self-attention
        and a position-wise feedforward network.

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
        self.ffn = PositionwiseFeedForward(hidden_dim, self.ffn_hidden_dim,
                                           dropout, activation)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout,
                                               batch_first=True)
        self.reset_parameters()

    def forward(self, x: Tensor, struct_mask: Optional[Tensor],
                key_pad: Tensor) -> Tensor:
        """Forward pass through transformer encoder layer.

        This method applies self-attention and a position-wise feedforward
        network to the input features.

        Args:
            x (Tensor): Padded node features
                (batch_size, max_nodes, hidden_dim)
            struct_mask (Optional[Tensor]): mask for structure-aware attention.
            key_pad (Tensor): Pre-computed key padding mask

        Returns:
            Tensor: Transformed features (batch_size, max_nodes, hidden_dim)
        """
        # self attention block
        additive = merge_masks(key_pad=key_pad, attn=struct_mask,
                               num_heads=self.num_heads, dtype=x.dtype,
                               detach=struct_mask is None)

        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=additive,
            need_weights=False,
        )
        x = residual + self.dropout_layer(attn_out)

        # feedforward block
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout_layer(ffn_out)

        return x

    def reset_parameters(self) -> None:
        """Reset parameters of the layer.

        This method resets the parameters of the self-attention,
        normalization layers, and the feedforward network.

        Args:
            None

        Returns:
            None
        """
        self.self_attn._reset_parameters()
        for module in [self.norm1, self.norm2, self.ffn]:
            reset(module)


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
        self.layers = nn.ModuleList([
            type(encoder_layer)(
                hidden_dim=encoder_layer.hidden_dim,
                num_heads=encoder_layer.num_heads,
                dropout=encoder_layer.dropout_layer.p,
                ffn_hidden_dim=encoder_layer.ffn_hidden_dim,
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.num_heads = encoder_layer.num_heads
        self.reset_parameters()

    def forward(self, x: Tensor, struct_mask: Optional[Tensor] = None,
                pad_cache: Optional[dict] = None):
        """Apply all encoder layers in sequence.

        If all layers share the same num_heads, build the pad once;
        if any layer has a different head‐count, rebuild it before that layer.

        Args:
            x (torch.Tensor): Node feature tensor
                (batch_size, max_nodes, hidden_dim)
            struct_mask (torch.Tensor, optional):Structure-aware attention mask
                (batch_size, num_heads, max_nodes, max_nodes)
            pad_cache (dict, optional): Pre-computed key padding masks for each
                number of heads. If None, it will be built from the input.

        Returns:
            torch.Tensor: Output after passing through all encoder layers.
        """
        for layer in self.layers:
            x = layer(
                x,
                struct_mask,
                pad_cache[layer.num_heads],
            )
        return x

    def reset_parameters(self) -> None:
        """Reset parameters of all layers.

        This method resets the parameters of all layers in the encoder.

        Args:
            None

        Returns:
            None
        """
        reset(self.layers)

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

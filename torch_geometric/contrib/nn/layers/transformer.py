import torch.nn as nn


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim):
        """Initialize a GraphTransformerEncoderLayer.

        Args:
            hidden_dim (int): Dimension of the input features
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)  # Normalization before attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # Normalization before FFN

    def forward(self, x, batch=None):
        """Forward pass that returns the input unchanged.

        Args:
            x (torch.Tensor): Node feature tensor
            batch (torch.Tensor, optional): Batch vector

        Returns:
            torch.Tensor: The input tensor unchanged
        """
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
        self.layers = nn.ModuleList([
            encoder_layer if i == 0 else type(encoder_layer)(
                encoder_layer.norm1.normalized_shape[0])
            for i in range(num_layers)
        ])
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

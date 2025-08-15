from typing import Callable

import torch


class QFormer(torch.nn.Module):
    r"""The Querying Transformer (Q-Former) from
    `"BLIP-2: Bootstrapping Language-Image Pre-training
    with Frozen Image Encoders and Large Language Models"
    <https://arxiv.org/pdf/2301.12597>`_ paper.

    Args:
        input_dim (int): The number of features in the input.
        hidden_dim (int): The dimension of the fnn in the encoder layer.
        output_dim (int): The final output dimension.
        num_heads (int): The number of multi-attention-heads.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        dropout (int): The dropout value in each encoder layer.


    .. note::
        This is a simplified version of the original Q-Former implementation.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.0,
            activation: Callable = torch.nn.ReLU(),
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )
        self.project = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Input sequence to the encoder layer.
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, sequence length :math:`N`,
                and feature dimension :math:`F`.
        """
        x = self.layer_norm(x)
        x = self.encoder(x)
        out = self.project(x)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_heads={self.num_heads}, '
                f'num_layers={self.num_layers})')

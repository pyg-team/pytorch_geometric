from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)


class SetTransformerAggregation(Aggregation):
    r"""Performs "Set Transformer" aggregation in which the elements to
    aggregate are processed by multi-head attention blocks, as described in
    the `"Graph Neural Networks with Adaptive Readouts"
    <https://arxiv.org/abs/2211.04952>`_ paper.

    Args:
        channels (int): Size of each input sample.
        num_seed_points (int, optional): Number of seed points.
            (default: :obj:`1`)
        num_encoder_blocks (int, optional): Number of Set Attention Blocks
            (SABs) in the encoder. (default: :obj:`1`).
        num_decoder_blocks (int, optional): Number of Set Attention Blocks
            (SABs) in the decoder. (default: :obj:`1`).
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the seed embeddings
            are averaged instead of concatenated. (default: :obj:`True`)
        norm (str, optional): If set to :obj:`True`, will apply layer
            normalization. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 1,
        heads: int = 1,
        concat: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.num_seed_points = num_seed_points
        self.heads = heads
        self.concat = concat
        self.layer_norm = layer_norm

        self.encoders = torch.nn.ModuleList([
            SetAttentionBlock(channels, heads, layer_norm)
            for _ in range(num_encoder_blocks)
        ])

        self.pma = PoolingByMultiheadAttention(channels, num_seed_points,
                                               heads, layer_norm)

        self.decoders = torch.nn.ModuleList([
            SetAttentionBlock(channels, heads, layer_norm)
            for _ in range(num_decoder_blocks)
        ])

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        x, mask = self.to_dense_batch(x, index, ptr, dim_size, dim)

        for encoder in self.encoders:
            x = encoder(x, mask)

        x = self.pma(x, mask)

        for decoder in self.decoders:
            x = decoder(x)

        return x.flatten(1, 2) if self.concat else x.mean(dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'num_seed_points={self.num_seed_points}, '
                f'heads={self.heads}, '
                f'layer_norm={self.layer_norm})')

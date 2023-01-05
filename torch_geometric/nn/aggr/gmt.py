from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)


class GraphMultisetTransformer(Aggregation):
    r"""The Graph Multiset Transformer pooling operator from the
    `"Accurate Learning of Graph Representations
    with Graph Multiset Pooling" <https://arxiv.org/abs/2102.11533>`_ paper.

    The :class:`GraphMultisetTransformer` aggregates elements into
    :math:`k` representative elements via attention-based pooling, computes the
    interaction among them via :obj:`num_encoder_blocks` self-attention blocks,
    and finally pools the representative elements via attention-based pooling
    into a single cluster.

    Args:
        channels (int): Size of each input sample.
        k (int): Number of :math:`k` representative nodes after pooling.
        num_encoder_blocks (int, optional): Number of Set Attention Blocks
            (SABs) between the two pooling blocks. (default: :obj:`1`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`True`, will apply layer
            normalization. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        k: int,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.k = k
        self.heads = heads
        self.layer_norm = layer_norm

        self.pma1 = PoolingByMultiheadAttention(channels, k, heads, layer_norm)
        self.encoders = torch.nn.ModuleList([
            SetAttentionBlock(channels, heads, layer_norm)
            for _ in range(num_encoder_blocks)
        ])
        self.pma2 = PoolingByMultiheadAttention(channels, 1, heads, layer_norm)

    def reset_parameters(self):
        self.pma1.reset_parameters()
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma2.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        x, mask = self.to_dense_batch(x, index, ptr, dim_size, dim)

        x = self.pma1(x, mask)

        for encoder in self.encoders:
            x = encoder(x)

        x = self.pma2(x)

        return x.squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'k={self.k}, heads={self.heads}, '
                f'layer_norm={self.layer_norm})')

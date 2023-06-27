from typing import Optional

import torch
from torch import Tensor

from torch_geometric.experimental import disable_dynamic_shapes
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

    .. note::

        :class:`GraphMultisetTransformer` requires sorted indices :obj:`index`
        as input. Specifically, if you use this aggregation as part of
        :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
        or by calling :meth:`torch_geometric.data.Data.sort`.

    Args:
        channels (int): Size of each input sample.
        k (int): Number of :math:`k` representative nodes after pooling.
        num_encoder_blocks (int, optional): Number of Set Attention Blocks
            (SABs) between the two pooling blocks. (default: :obj:`1`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`True`, will apply layer
            normalization. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """
    def __init__(
        self,
        channels: int,
        k: int,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.k = k
        self.heads = heads
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.pma1 = PoolingByMultiheadAttention(channels, k, heads, layer_norm,
                                                dropout)
        self.encoders = torch.nn.ModuleList([
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_encoder_blocks)
        ])
        self.pma2 = PoolingByMultiheadAttention(channels, 1, heads, layer_norm,
                                                dropout)

    def reset_parameters(self):
        self.pma1.reset_parameters()
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma2.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        x, mask = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                      max_num_elements=max_num_elements)

        x = self.pma1(x, mask)

        for encoder in self.encoders:
            x = encoder(x)

        x = self.pma2(x)

        return x.squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'k={self.k}, heads={self.heads}, '
                f'layer_norm={self.layer_norm}, '
                f'dropout={self.dropout})')

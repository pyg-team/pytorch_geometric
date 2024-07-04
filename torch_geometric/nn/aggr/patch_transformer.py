from typing import Optional

import torch
from torch import Tensor

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.pt_utils import (
    FeatEncoder,
    PositionalEncoding1D,
    TransformerBlock,
)
from torch_geometric.nn.encoding import TemporalEncoding


class PatchTransformerAggregation(Aggregation):
    r"""Performs  Heterogeneous Link Encoder embedding
    and Semantic Patches Fusion
    from `"Simplifying Temporal Heterogeneous Network for
    Continuous-Time Link Prediction
    "<https://dl.acm.org/doi/pdf/10.1145/3583780.3615059>`_paper.
    .. note::
        :class:`PatchTransformerAggregation` requires indices :obj:`index`,
        edge features :obj:`edge_feats` and edge timestamps :obj:`edge_ts`
        as input.
        The output dimension must be fixed
        and can be set via `out_dim` argument.

    Args:
        channels (int): Size of each input sample, node embedding size
        edge_channels (int): edge embedding size.
        time_dim (int, optional): time embedding dimension,
        use to specify the time embedding dimension of edge timestamps
        hidden_dim (int, optional): hidden dimension
        out_dim (int, optional): output dimension
        num_encoder_blocks (int, optional): Number of Set Attention Blocks
            (SABs) in the encoder. (default: :obj:`1`).
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        patch_size (int, optional): Number of links in a patch
            (default: :obj:`5`)
        norm (str, optional): If set to :obj:`True`, will apply layer
            normalization. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """
    def __init__(
        self,
        channels: int,
        edge_channels: int,
        time_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_edge_per_node: int = 64,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        patch_size: int = 5,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        # node dim
        self.channels = channels
        self.edge_channels = edge_channels
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads
        self.layer_norm = layer_norm
        self.patch_size = patch_size
        self.dropout = dropout
        self.num_edge_per_node = num_edge_per_node

        # encoders and linear layers
        self.feat_encoder = FeatEncoder(self.time_dim, self.edge_channels,
                                        self.hidden_dim)
        self.layernorm = torch.nn.LayerNorm(self.hidden_dim)
        self.time_encoder = TemporalEncoding(self.time_dim)
        self.atten_blocks = torch.nn.ModuleList([
            TransformerBlock(dims=self.hidden_dim, heads=self.heads,
                             dropout=self.dropout)
            for _ in range(num_encoder_blocks)
        ])

        # input is node feature and transformer node embed
        self.mlp_head = torch.nn.Linear(self.hidden_dim + self.channels,
                                        self.out_dim)

        # padding
        self.stride = self.patch_size
        self.pad_projector = torch.nn.Linear(self.patch_size * self.hidden_dim,
                                             self.hidden_dim)
        self.pe = PositionalEncoding1D(self.hidden_dim)

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
        self.pma.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Tensor,
        edge_feats: Tensor,
        edge_ts: Tensor,
        batch_size: int,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x_feat = torch.zeros(
            (batch_size * self.num_edge_per_node, edge_time_feats.size(1)),
            device=edge_feats.device)

        # need to check what is ids shape
        x_feat[index] = x_feat[index] + edge_time_feats
        x_feat = x_feat.view(-1, self.num_edge_per_node // self.patch_size,
                             self.patch_size * x_feat.shape[-1])

        # sum operation
        x_feat = self.pad_projector(x_feat)
        x_feat = x_feat + self.pe(x_feat)

        for atten_block in self.atten_blocks:
            x_feat = atten_block(x_feat)

        x_feat = self.layernorm(x_feat)
        x_feat = torch.mean(x_feat, dim=1)
        x = torch.cat([x, x_feat], dim=1)
        return self.mlp_head(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'edge_channels={self.edge_channels},'
                f'time_dim={self.time_dim},'
                f'heads={self.heads},'
                f'layer_norm={self.layer_norm},'
                f'dropout={self.dropout})')

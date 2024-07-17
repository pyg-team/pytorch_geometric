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


class PatchTransformerAggregation(Aggregation):
    r"""Performs  Heterogeneous Link Encoder embedding
    and Semantic Patches Fusion
    from `"Simplifying Temporal Heterogeneous Network for
    Continuous-Time Link Prediction
    "<https://dl.acm.org/doi/pdf/10.1145/3583780.3615059>`_paper.

    Args:
        channels (int): Size of each edge feature
        use to specify the time embedding dimension of edge timestamps
        hidden_dim (int, optional): hidden dimension
        out_dim (int, optional): output dimension
        max_edge (int, optional): max number of edges per node
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
        aggr (list[str], optional): aggregation module, ['sum','max','mean']
        time_dim (int, optional): time embedding dimension,

    """
    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        max_edge: int = 16,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        patch_size: int = 5,
        layer_norm: bool = False,
        dropout: float = 0.0,
        aggr: list[str] = None,
        time_dim: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads
        self.layer_norm = layer_norm
        self.patch_size = patch_size
        self.dropout = dropout
        self.max_edge = max_edge
        self.aggr = aggr
        if (self.aggr is None):
            self.aggr = ["mean"]

        # encoders and linear layers
        self.feat_encoder = FeatEncoder(self.channels, self.hidden_dim,
                                        time_dim=self.time_dim)
        self.layernorm = torch.nn.LayerNorm(self.hidden_dim)
        self.atten_blocks = torch.nn.ModuleList([
            TransformerBlock(dims=self.hidden_dim, heads=self.heads,
                             dropout=self.dropout)
            for _ in range(num_encoder_blocks)
        ])

        # input is node feature and transformer node embed
        self.mlp_head = torch.nn.Linear(self.hidden_dim * (len(self.aggr)),
                                        self.out_dim)

        # padding
        self.stride = self.patch_size
        self.pad_projector = torch.nn.Linear(self.patch_size * self.hidden_dim,
                                             self.hidden_dim)
        self.pe = PositionalEncoding1D(self.hidden_dim)

    def reset_parameters(self):
        self.feat_encoder.reset_parameters()
        for atten_block in self.atten_blocks:
            atten_block.reset_parameters()
        self.mlp_head.reset_parameters()
        self.pad_projector.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        # x = [num_edges, input_channels]
        x = self.feat_encoder(x)

        # x = [num_edges, hidden_dim]
        edge_feat_size = x.shape[1]

        # x = [batch_size, max_num_elements, -1]
        x, mask = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                      max_num_elements=max_num_elements)

        x_feat = x.view(mask.shape[0] * max_num_elements, edge_feat_size)
        # x_feat = [batch_size, # patch, patch_size * edge_feat_size]
        x_feat = x_feat.view(-1, self.max_edge // self.patch_size,
                             self.patch_size * x_feat.shape[-1])
        # x_feat = [batch_size, max_edge, edge_feat_size]
        x_feat = self.pad_projector(x_feat)

        # x_feat = [batch_size, max_edge, edge_feat_size]
        x_feat = x_feat + self.pe(x_feat)

        # x_feat = [batch_size, max_edge, edge_feat_size]
        for atten_block in self.atten_blocks:
            x_feat = atten_block(x_feat)

        x_feat = self.layernorm(x_feat)

        # aggregate with a list of operations
        # x_feat, _ = torch.max(x_feat, dim=1)

        out_list = []
        for aggr in self.aggr:
            if aggr == "sum":
                out_list.append(torch.sum(x_feat, dim=1))
            elif aggr == "mean":
                out_list.append(torch.mean(x_feat, dim=1))
            elif aggr == "max":
                out, _ = torch.max(x_feat, dim=1)
                out_list.append(out)
            else:
                raise ValueError(f"the following aggregation is not supported"
                                 f"'{aggr}'")

        x_final = torch.cat(out_list, dim=1)

        return self.mlp_head(x_final)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels},'
                f'heads={self.heads},'
                f'layer_norm={self.layer_norm},'
                f'aggr={self.aggr},'
                f'dropout={self.dropout})')

from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.gmt import PMA, SAB


class SetTransformer(torch.nn.Module):
    def __init__(self, dim_input: int, num_outputs: int, dim_output: int,
                 num_enc_SABs: int, num_dec_SABs: int, dim_hidden: int = 128,
                 num_heads: int = 4, ln: bool = False):
        super().__init__()

        self.enc = torch.nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, layer_norm=ln), *[
                SAB(dim_hidden, dim_hidden, num_heads, layer_norm=ln)
                for _ in range(num_enc_SABs)
            ])

        self.dec = torch.nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, layer_norm=ln), *[
                SAB(dim_hidden, dim_hidden, num_heads, layer_norm=ln)
                for _ in range(num_dec_SABs)
            ], torch.nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class SetTransformerAggregation(Aggregation):
    r"""Performs Set Transformer aggregation in which the graph nodes are
    considered to be set elements processed by multi-head attention blocks, as
    described in the `"Graph Neural Networks with Adaptive Readouts"
    <https://arxiv.org/abs/2211.04952>`_ paper. SetTransformerAggregation is a
    permutation-invariant operator.

    Args:
        max_num_nodes_in_dataset (int): Maximum number of nodes in a graph in
            the dataset.
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_PMA_outputs (int): Number of outputs for the PMA block.
        num_enc_SABs (int): Number of SAB blocks to use in the encoder after
            the initial one that is always present (can be zero).
        num_dec_SABs (int): Number of SAB blocks to use in the decoder
            (can be zero).
        dim_hidden (int): Hidden dimension used inside the attention blocks.
        num_heads (int): Number of heads for the multi-head attention blocks.
        ln (bool): Whether to use layer normalisation in the attention blocks.
    """
    def __init__(self, max_num_nodes_in_dataset: int, in_channels: int,
                 out_channels: int, num_PMA_outputs: int, num_enc_SABs: int,
                 num_dec_SABs: int, dim_hidden: int, num_heads: int,
                 ln: bool = False):
        super().__init__()

        self.max_num_nodes_in_dataset = max_num_nodes_in_dataset
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_PMA_outputs = num_PMA_outputs
        self.num_enc_SABs = num_enc_SABs
        self.num_dec_SABs = num_dec_SABs
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.ln = ln

        self.set_transformer_agg = SetTransformer(in_channels, num_PMA_outputs,
                                                  out_channels, num_enc_SABs,
                                                  num_dec_SABs, dim_hidden,
                                                  num_heads, ln)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.set_transformer_agg.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(
            x, index, ptr, dim_size, dim,
            max_num_elements=self.max_num_nodes_in_dataset)
        x = self.set_transformer_agg(x)
        return x.mean(dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.max_num_nodes_in_dataset}, '
                f'{self.in_channels}, {self.out_channels}, '
                f'{self.num_PMA_outputs}, {self.num_enc_SABs}, '
                f'{self.num_dec_SABs}, {self.dim_hidden}, '
                f'{self.num_heads}, {self.ln})')

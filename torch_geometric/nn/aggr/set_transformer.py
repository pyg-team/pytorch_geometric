import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.aggr import Aggregation

# Set Transformer code as provided in
# https://github.com/juho-lee/set_transformer


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(
            Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        Out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        Out = Out if getattr(self, 'ln0', None) is None else self.ln0(Out)
        Out = Out + F.relu(self.fc_o(Out))
        Out = Out if getattr(self, 'ln1', None) is None else self.ln1(Out)
        return Out


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.Induced = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.Induced)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.Induced.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input: int, num_outputs: int, dim_output: int,
                 num_enc_SABs: int, num_dec_SABs: int, dim_hidden: int = 128,
                 num_heads: int = 4, ln: bool = False):
        super(SetTransformer, self).__init__()

        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln), *[
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
                for _ in range(num_enc_SABs)
            ])

        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln), *[
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
                for _ in range(num_dec_SABs)
            ], nn.Linear(dim_hidden, dim_output))

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

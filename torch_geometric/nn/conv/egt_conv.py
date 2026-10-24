from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Sequential

from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch


class EGTConv(torch.nn.Module):
    r"""The Edge-augmented Graph Transformer (EGT) from the
    `"Global Self-Attention as a Replacement for Graph Convolution"
    <https://arxiv.org/abs/2108.03348>`_ paper.

    Args:
        channels (int): Size of each input sample.
        edge_dim (int): Edge feature dimensionality.
        num_virtual_nodes (int): Number of virtual nodes.
        edge_update (Bool, optional): Whether to update the edge embedding.
            (default: :obj:`True`)
        heads (int, optional): Number of attention heads,
            by which :attr:`channels` is divisible. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        attn_dropout (float, optional): Attention dropout probability
            of intermediate embeddings. (default: :obj:`0.`)
    """
    def __init__(
        self,
        channels: int,
        edge_dim: int,
        num_virtual_nodes: int,
        edge_update: bool = True,
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'elu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        assert channels % heads == 0, "channels must be divisible by heads"

        self.channels = channels
        self.num_virtual_nodes = num_virtual_nodes
        self.heads = heads
        self.edge_update = edge_update
        self.hidden_channels = channels // heads
        self.mha_ln_h = LayerNorm(channels)
        self.mha_ln_e = LayerNorm(edge_dim)
        self.mha_dropout_h = Dropout(dropout)
        self.edge_input = Linear(edge_dim, heads)
        self.qkv_proj = Linear(channels, channels * 3)
        self.gate = Linear(edge_dim, heads)
        self.attn_dropout = Dropout(attn_dropout)
        self.node_output = Linear(channels, channels)

        self.node_ffn = Sequential(
            LayerNorm(channels),
            Linear(channels, channels),
            activation_resolver(act, **(act_kwargs or {})),
            Linear(channels, channels),
            Dropout(dropout),
        )

        if self.edge_update:
            self.edge_output = Linear(heads, edge_dim)
            self.mha_dropout_e = Dropout(dropout)
            self.edge_ffn = Sequential(
                LayerNorm(edge_dim), Linear(edge_dim, edge_dim),
                activation_resolver(act, **(act_kwargs or {})),
                Linear(edge_dim, edge_dim), Dropout(dropout))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mha_ln_h.reset_parameters()
        self.mha_ln_e.reset_parameters()
        self.edge_input.reset_parameters()
        self.qkv_proj.reset_parameters()
        self.gate.reset_parameters()
        self.node_output.reset_parameters()
        reset(self.node_ffn)
        if self.edge_update:
            self.edge_output.reset_parameters()
            reset(self.edge_ffn)

    def forward(
        self,
        x: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        h, mask = to_dense_batch(x, batch)  # [B, N, channels]
        e = edge_attr  # [B, N, N, edge_dim]

        h_ln = self.mha_ln_h(h)  # [B, N, channels]
        e_ln = self.mha_ln_e(e)  # [B, N, N, edge_dim]
        qkv = self.qkv_proj(h_ln)  # [B, N, channels * 3]
        e_bias = self.edge_input(e_ln)  # [B, N, N, heads]
        gates = self.gate(e_ln)  # [B, N, N, heads]
        B, N, _ = qkv.size()
        q_h, k_h, v_h = qkv.view(B, N, -1, self.heads).split(
            self.hidden_channels,
            dim=2)  # each should be [B, N, hidden_channels, heads]
        attn_hat = torch.einsum("bldh,bmdh->blmh", q_h, k_h)
        attn_hat = attn_hat.clamp(-5, 5) + e_bias  # [B, N, N, heads]

        gates = F.sigmoid(gates)  # [B, N, N, heads]
        attn_tild = F.softmax(attn_hat, dim=2) * gates  # [B, N, N, heads]
        attn_tild = self.attn_dropout(attn_tild)

        v_attn = torch.einsum("blmh,bmkh->blkh", attn_tild, v_h)

        # Scale the aggregated values by degree.
        degrees = torch.sum(gates, dim=2, keepdim=True)  # [B, N, 1, heads]
        degrees_scalers = torch.log(1 + degrees)
        degrees_scalers[:, :self.num_virtual_nodes] = 1.0
        v_attn = v_attn * degrees_scalers
        v_attn = v_attn.reshape(B, N, -1)  # [B, N, channels]

        out1_h = self.mha_dropout_h(self.node_output(v_attn)) + h
        out2_h = self.node_ffn(out1_h) + out1_h  # [B, N, channels]

        if self.edge_update:
            out1_e = self.mha_dropout_e(self.edge_output(attn_hat)) + e
            out2_e = self.edge_ffn(out1_e) + out1_e  # [B, N, N, edge_dim]
            return out2_h[mask], out2_e

        return out2_h[mask]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.channels}, '
            f'heads={self.heads}, num_virtual_nodes={self.num_virtual_nodes})')

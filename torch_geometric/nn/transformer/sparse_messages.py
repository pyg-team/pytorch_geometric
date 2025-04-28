from typing import Optional

import torch
from xformers.components.attention.core import (
    SparseCS,
    _apply_dropout,
    _matmul_with_mask,
    _softmax,
    bmm,
    scaled_query_key_softmax,
)

from torch_geometric.nn import MessagePassing


class Kronecker(MessagePassing):
    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = dropout

    def forward(self, key: torch.Tensor, query: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor,
                B: int) -> torch.Tensor:
        return self.message(key, query, value, mask, B)

    def message(self, key: torch.Tensor, query: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor,
                B: int) -> torch.Tensor:
        out = []
        for head_idx in range(mask.shape[0]):
            head_mask = torch.stack([mask[head_idx] for _ in range(B)], dim=0)
            head_mask = SparseCS(head_mask, query.device)  #(N,N)

            att = _matmul_with_mask(query[:, head_idx, :, :],
                                    key.transpose(-2, -1)[:, head_idx, :, :],
                                    head_mask)
            att = _softmax(att, causal=False)  #B,N,N # edge_softmax
            att = _apply_dropout(att, self.dropout)
            context_layer = bmm(att, value[:, head_idx, :, :])  # B,N,Dh
            out.append(context_layer)
        return torch.stack(out, dim=1), att

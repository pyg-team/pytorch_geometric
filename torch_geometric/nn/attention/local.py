from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


class LocalAttention(MessagePassing):
    """Local neighborhood attention."""
    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:

        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                "edge_attr size doesn't match the no of edges in edge_index.")
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        edge_attention = self.edge_proj(
            edge_attr) if edge_attr is not None else None
        out = self.propagate(edge_index, q=q, k=k, v=v,
                             edge_attr=edge_attention)
        return self.o_proj(out.view(-1, self.hidden_dim))

    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor]) -> torch.Tensor:

        attention = (q_i * k_j).sum(dim=-1) / np.sqrt(self.head_dim)
        if edge_attr is not None:

            if edge_attr.size(0) < attention.size(0):
                num_repeats = attention.size(0) // edge_attr.size(0) + 1
                edge_attr = edge_attr.repeat(num_repeats,
                                             1)[:attention.size(0)]
            elif edge_attr.size(0) > attention.size(0):
                edge_attr = edge_attr[:attention.size(0)]
            attention = attention + edge_attr

        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # Apply attention scores to the values
        out = attention.unsqueeze(-1) * v_j
        return out

    def edge_update(self) -> torch.Tensor:
        raise NotImplementedError(
            "edge_update not implemented in LocalAttention.")

    def message_and_aggregate(self, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "message_and_aggregate not implemented in LocalAttention.")

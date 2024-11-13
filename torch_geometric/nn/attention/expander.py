import torch
import torch.nn as nn
import numpy as np

from typing import Tuple
from torch_geometric.nn import MessagePassing


class ExpanderAttention(MessagePassing):
    """Expander graph attention using random d-regular near-Ramanujan graphs."""
    def __init__(self, hidden_dim: int, expander_degree: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.expander_degree = expander_degree
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embedding = nn.Parameter(torch.randn(1, num_heads))
        self.dropout = nn.Dropout(dropout)

    def generate_expander_edges(self, num_nodes: int) -> torch.Tensor:
        edges = []
        for _ in range(self.expander_degree // 2):
            perm = torch.randperm(num_nodes)
            edges.extend([(i, perm[i].item()) for i in range(num_nodes)])
            edges.extend([(perm[i].item(), i) for i in range(num_nodes)])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index

    def forward(self, x: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = self.generate_expander_edges(num_nodes).to(x.device)
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        out = self.propagate(edge_index, q=q, k=k, v=v)
        return self.o_proj(out.view(-1, self.hidden_dim)), edge_index

    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor) -> torch.Tensor:
        attention = (q_i * k_j).sum(dim=-1) / np.sqrt(self.head_dim)
        attention = torch.softmax(attention + self.edge_embedding, dim=-1)
        attention = self.dropout(attention)
        return attention.unsqueeze(-1) * v_j
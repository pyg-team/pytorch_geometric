import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import (
    GINConv,
    GINEConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_dense_batch
from typing import Optional, Tuple, Union

class MultiheadAttention(nn.Module):
    def __init__(self, heads: int, dim_model: int, dropout: float = 0.1):
        super().__init__()
        assert dim_model % heads == 0
        self.d_k = dim_model // heads
        self.heads = heads
        self.linears = ModuleList([Linear(dim_model, dim_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, sph, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores * sph
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, sph, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key, value = [
            linear(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(query, key, value, sph, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        return self.linears[-1](x)

class GatedGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__(aggr='add', **kwargs)
        self.A = Linear(in_channels, out_channels, bias=True)
        self.B = Linear(in_channels, out_channels, bias=True)
        self.C = Linear(in_channels, out_channels, bias=True)
        self.D = Linear(in_channels, out_channels, bias=True)
        self.E = Linear(in_channels, out_channels, bias=True)
        
        self.bn_node_x = nn.BatchNorm1d(out_channels)
        self.bn_edge_e = nn.BatchNorm1d(out_channels)
        self.act_fn_x = ReLU()
        self.act_fn_e = ReLU()
        self.dropout = dropout
        self.residual = residual

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        pe: OptTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.residual:
            x_in = x
            edge_attr_in = edge_attr

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(edge_attr) if edge_attr is not None else None
        Dx = self.D(x)
        Ex = self.E(x)

        x_out, edge_attr_out = self.propagate(
            edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, x=x, Ax=Ax, edge_attr=edge_attr
        )

        x_out = self.bn_node_x(x_out)
        edge_attr_out = self.bn_edge_e(edge_attr_out)

        x_out = self.act_fn_x(x_out)
        edge_attr_out = self.act_fn_e(edge_attr_out)

        x_out = F.dropout(x_out, self.dropout, training=self.training)
        edge_attr_out = F.dropout(edge_attr_out, self.dropout, training=self.training)

        if self.residual:
            x_out = x_in + x_out
            edge_attr_out = edge_attr_in + edge_attr_out

        return x_out, edge_attr_out

    def message(self, Dx_i, Ex_j, Ce: OptTensor = None) -> torch.Tensor:
        e_ij = Dx_i + Ex_j
        if Ce is not None:
            e_ij = e_ij + Ce
        return torch.sigmoid(e_ij)

    def aggregate(self, sigma_ij: torch.Tensor, index: torch.Tensor, Bx_j: torch.Tensor,
                 size_i: Optional[int] = None) -> torch.Tensor:
        numerator = torch.zeros_like(Bx_j)
        numerator.scatter_add_(0, index.view(-1, 1).expand(-1, Bx_j.size(-1)), sigma_ij * Bx_j)
        
        denominator = torch.zeros_like(Bx_j)
        denominator.scatter_add_(0, index.view(-1, 1).expand(-1, Bx_j.size(-1)), sigma_ij)
        
        out = numerator / (denominator + 1e-6)
        return out

class Gradformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.0,
        pe_dim: int = 16,
        pool: str = 'mean',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool
        
        self.pe_lin = Linear(pe_dim, hidden_channels)
        self.node_lin = Linear(in_channels, hidden_channels - pe_dim)
        
        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = GatedGCNConv(
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                residual=True,
            )
            self.convs.append(conv)

        self.attention = MultiheadAttention(
            heads=heads,
            dim_model=hidden_channels,
            dropout=dropout,
        )

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, hidden_channels // 4),
            ReLU(),
            Linear(hidden_channels // 4, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        pe: torch.Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        sph: OptTensor = None,
    ) -> torch.Tensor:
        x = torch.cat([self.node_lin(x), self.pe_lin(pe)], dim=-1)

        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_attr, pe)
            
            # Global attention
            x_dense, mask = to_dense_batch(x, batch)
            x_dense = self.attention(x_dense, x_dense, x_dense, sph, mask=~mask)
            x = x_dense[mask]
            
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        if batch is not None:
            if self.pool == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pool == 'add':
                x = global_add_pool(x, batch)
            elif self.pool == 'max':
                x = global_max_pool(x, batch)

        return self.mlp(x) 


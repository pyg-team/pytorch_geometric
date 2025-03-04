import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch

"""

This script contains separate tests for:
  1. The Attention module with an exponential decay mask (including MultiheadAttention).
  2. The GatedGCNLayer (simplified version for message passing).
  3. The GPSConv module that fuses local MPNN with global self-attention.
  4. The full Gradformer model.

Full paper <https://arxiv.org/abs/2404.15729>
"""

# ===============================
# 1. Attention Module with Exponential Decay Mask
# ===============================


def clones(module, N):
    """Produce N identical layers."""
    # This function creates a list of N identical layers using deep copies
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, sph, mask=None, dropout=None):
    """Compute Scaled Dot-Product Attention with an exponential decay mask.

    Args:
        query, key, value: Input tensors for attention mechanism.
        sph: Exponential decay mask based on node distances.
        mask: Optional mask for padding or invalid positions.
        dropout: Optional dropout for regularization.

    Returns:
        Output tensor after attention is applied.
    """
    d_k = query.size(-1)

    print(f"Query shape: {query.shape}"
          )  # Expected: (batch_size, h, seq_length, d_k)
    print(f"Key^T shape: {key.transpose(-2, -1).shape}"
          )  # Expected: (batch_size, h, seq_length, d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    print(f"Attention scores shape: {scores.shape}"
          )  # Expected: (batch_size, h, seq_length, seq_length)
    print(f"Exponential decay mask (sph) shape before expansion: {sph.shape}"
          )  # Expected: (batch_size, seq_length, seq_length)

    print(f"Exponential decay mask (sph) shape after expansion: {sph.shape}")

    scores = scores * sph

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)

    return output, p_attn


class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Multi-head attention module.

        Args:
            h: Number of attention heads.
            d_model: Dimension of the input and output features.
            dropout: Dropout probability.
        """
        super().__init__()

        assert d_model % h == 0, "d_model must be divisible by the number of heads"

        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, sph, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = (l(x) for l, x in zip(self.linears, (query, key,
                                                                 value)))

        query, key, value = (x.view(batch_size, -1, self.h,
                                    self.d_k).transpose(1, 2)
                             for x in (query, key, value))

        x, self.attn = attention(query, key, value, sph, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.h * self.d_k)

        return self.linears[-1](x)


def test_attention_module():
    print("=== Testing MultiheadAttention Module ===")
    batch_size = 2  # Number of sequences in a batch
    seq_length = 5  # Length of each sequence
    d_model = 16  # Model dimension (must be divisible by number of heads)
    h = 4  # Number of attention heads

    # creating random input tensors for query, key, and value.
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)

    # IMPORTANT: Provide the exponential decay mask (sph) with shape [B, h, S, S]
    # because the core code multiplies "scores" (shape [B, h, S, S]) by sph.
    sph = torch.ones(batch_size, h, seq_length, seq_length)

    mha = MultiheadAttention(h, d_model)

    output = mha(query, key, value, sph)

    # Print the output shape (should be [batch_size, seq_length, d_model])
    print(f"Output shape from MultiheadAttention: {output.shape}")
    print()


# ===============================
# 2. GatedGCN Layer (Simplified)
# ===============================


class GatedGCNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, residual=True):
        """A simplified GatedGCN layer.
        """
        super().__init__(aggr='add')
        self.A = nn.Linear(in_dim, out_dim)
        self.B = nn.Linear(in_dim, out_dim)
        # Additional layers for demonstration (C, D, E)
        self.C = nn.Linear(in_dim, out_dim)
        self.D = nn.Linear(in_dim, out_dim)
        self.E = nn.Linear(in_dim, out_dim)
        self.bn_node = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, edge_index, edge_attr=None):
        x_in = x
        Ax = self.A(x)
        Bx = self.B(x)
        out = self.propagate(edge_index, x=Bx)
        out = Ax + out
        if self.residual:
            out = out + x_in
        out = self.bn_node(out)
        return out

    def message(self, x_j):
        return x_j


def test_gatedgcn_layer():
    print("=== Testing GatedGCNLayer ===")
    # Dummy node features for 10 nodes, dimension 16
    x = torch.rand(10, 16)
    # Create a simple chain graph (edges from node i to i+1)
    edge_index = torch.tensor([[i
                                for i in range(9)], [i + 1 for i in range(9)]],
                              dtype=torch.long)
    layer = GatedGCNLayer(in_dim=16, out_dim=16, dropout=0.1)
    out = layer(x, edge_index)
    print("GatedGCNLayer output shape:", out.shape)
    print()


# ===============================
# 3. GPSConv Module (Simplified)
# ===============================


class GPSConv(nn.Module):
    def __init__(self, channels, conv, heads=4, dropout=0.0):
        """GPSConv fuses a local message passing layer with a global self-attention layer.
        """
        super().__init__()
        self.conv = conv
        self.attn = MultiheadAttention(heads, channels, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x, edge_index, sph, batch):
        # Local message passing
        if self.conv is not None:
            h = self.conv(x, edge_index)
            h = F.dropout(h, p=0.0, training=self.training)
            h = h + x
            h = self.norm1(h)
        else:
            h = x
        # Global attention – converting to a dense batch
        h_dense, mask = to_dense_batch(h, batch)
        h_attn = self.attn(h_dense, h_dense, h_dense, sph, mask=~mask)
        h_attn = h_attn[mask]
        h = h_attn + x
        h = self.norm2(h)
        out = h + self.mlp(h)
        return out


def test_gpsconv_module():
    print("=== Testing GPSConv Module ===")
    # Dummy features for 10 nodes, dimension 16
    x = torch.rand(10, 16)
    edge_index = torch.tensor([[i
                                for i in range(9)], [i + 1 for i in range(9)]],
                              dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)  # All nodes belong to one graph
    sph = torch.ones(10, 10)  # Dummy sph mask (ones)
    conv = GatedGCNLayer(in_dim=16, out_dim=16, dropout=0.1)
    gpsconv = GPSConv(channels=16, conv=conv, heads=4, dropout=0.1)
    out = gpsconv(x, edge_index, sph, batch)
    print("GPSConv output shape:", out.shape)
    print()


# ===============================
# 4. Gradformer Model
# ===============================


class Gradformer(nn.Module):
    def __init__(self, num_layers, channels, nhead, dropout, mpnn_type='GCN'):
        """Gradformer: A graph transformer stacking multiple GPSConv layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = GatedGCNLayer(channels, channels, dropout=dropout,
                                 residual=True)
            gps_conv = GPSConv(channels, conv, heads=nhead, dropout=dropout)
            self.convs.append(gps_conv)
        self.pool = global_mean_pool  # Global pooling
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2,
                      1)  # For regression or binary classification
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.size(0)
        # Dummy sph mask: in practice compute using process_hop() or similar
        sph = torch.ones((num_nodes, num_nodes), device=x.device)
        for conv in self.convs:
            x = conv(x, edge_index, sph, batch)
        x = self.pool(x, batch)
        out = self.mlp(x)
        return out


def test_gradformer_model():
    print("=== Testing Gradformer Model ===")
    num_nodes = 20
    x = torch.rand(num_nodes, 16)
    # Create a simple chain graph for demonstration
    edge_index = torch.tensor([[i for i in range(num_nodes - 1)],
                               [i + 1 for i in range(num_nodes - 1)]],
                              dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    # Dummy target for regression task
    y = torch.rand(1)
    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
    model = Gradformer(num_layers=2, channels=16, nhead=4, dropout=0.1,
                       mpnn_type='GCN')
    out = model(data)
    print("Gradformer output shape:", out.shape)
    print()


if __name__ == '__main__':
    test_attention_module()
    test_gatedgcn_layer()
    test_gpsconv_module()
    test_gradformer_model()
    print("All tests passed!")

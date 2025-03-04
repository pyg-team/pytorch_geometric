#!/usr/bin/env python
"""
Test Script for Gradformer Submodules

This script contains separate tests for:
  1. The Attention module with an exponential decay mask (including
     MultiheadAttention).
  2. The GatedGCNLayer (simplified version for message passing).
  3. The GPSConv module that fuses local MPNN with global self-attention.
  4. The full Gradformer model.

Full paper: <https://arxiv.org/abs/2404.15729>
"""

import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.data import Data, DataLoader, TUDataset
from torch_geometric.utils import to_dense_batch


# ===============================
# 1. Attention Module with Exponential Decay Mask
# ===============================

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, sph, mask=None, dropout=None):
    """
    Compute scaled dot-product attention with an exponential decay mask.

    Args:
        query (Tensor): Query tensor of shape 
            [batch_size, h, seq_length, d_k].
        key (Tensor): Key tensor of shape 
            [batch_size, h, seq_length, d_k].
        value (Tensor): Value tensor of shape 
            [batch_size, h, seq_length, d_k].
        sph (Tensor): Exponential decay mask of shape 
            [batch_size, h, seq_length, seq_length].
        mask (Tensor, optional): Mask for padding/invalid positions.
        dropout (nn.Dropout, optional): Dropout module.

    Returns:
        Tuple[Tensor, Tensor]: Output tensor and attention weights.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print(f"Query shape: {query.shape}")
    print(f"Key^T shape: {key.transpose(-2, -1).shape}")
    print(f"Attention scores shape: {scores.shape}")
    print(f"Exponential decay mask (sph) shape before expansion: {sph.shape}")
    # Note: In the core paper code, sph is assumed to already have the
    # proper shape ([B, h, S, S]). Therefore, we do not change it here.
    print(f"Exponential decay mask (sph) shape after expansion: {sph.shape}")

    scores = scores * sph
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Multi-head attention module.

        Args:
            h (int): Number of attention heads.
            d_model (int): Dimension of input and output features.
            dropout (float): Dropout probability.
        """
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by the number of heads"
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, sph, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # Linear projections for Q, K, V.
        query, key, value = [
            linear(x) for linear, x in zip(self.linears, (query, key, value))
        ]
        query, key, value = [
            x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for x in (query, key, value)
        ]
        x, self.attn = attention(query, key, value, sph, mask=mask,
                                  dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.h * self.d_k)
        return self.linears[-1](x)


def test_attention_module():
    print("=== Testing MultiheadAttention Module ===")
    batch_size = 2
    seq_length = 5
    d_model = 16
    h = 4
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)
    # Provide the exponential decay mask with shape [B, h, S, S]
    sph = torch.ones(batch_size, h, seq_length, seq_length)
    mha = MultiheadAttention(h, d_model)
    output = mha(query, key, value, sph)
    print(f"Output shape from MultiheadAttention: {output.shape}")
    print()


# ===============================
# 2. GatedGCN Layer (Simplified)
# ===============================

class GatedGCNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, residual=True):
        """
        A simplified GatedGCN layer.

        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            dropout (float): Dropout probability.
            residual (bool): Whether to include a residual connection.
        """
        super(GatedGCNLayer, self).__init__(aggr="add")
        self.A = nn.Linear(in_dim, out_dim)
        self.B = nn.Linear(in_dim, out_dim)
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
    x = torch.rand(10, 16)
    edge_index = torch.tensor(
        [[i for i in range(9)], [i + 1 for i in range(9)]],
        dtype=torch.long,
    )
    layer = GatedGCNLayer(in_dim=16, out_dim=16, dropout=0.1)
    out = layer(x, edge_index)
    print(f"GatedGCNLayer output shape: {out.shape}")
    print()


# ===============================
# 3. GPSConv Module (Simplified)
# ===============================

class GPSConv(nn.Module):
    def __init__(self, channels, conv, heads=4, dropout=0.0):
        """
        GPSConv fuses a local message passing layer with a global self-attention
        layer.

        Args:
            channels (int): Feature dimension.
            conv (nn.Module): Local message passing layer.
            heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super(GPSConv, self).__init__()
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
        if self.conv is not None:
            h = self.conv(x, edge_index)
            h = F.dropout(h, p=0.0, training=self.training)
            h = h + x
            h = self.norm1(h)
        else:
            h = x
        h_dense, mask = to_dense_batch(h, batch)
        h_attn = self.attn(h_dense, h_dense, h_dense, sph, mask=~mask)
        h_attn = h_attn[mask]
        h = h_attn + x
        h = self.norm2(h)
        out = h + self.mlp(h)
        return out


def test_gpsconv_module():
    print("=== Testing GPSConv Module ===")
    x = torch.rand(10, 16)
    edge_index = torch.tensor(
        [[i for i in range(9)], [i + 1 for i in range(9)]],
        dtype=torch.long,
    )
    batch = torch.zeros(10, dtype=torch.long)
    sph = torch.ones(10, 10)
    conv = GatedGCNLayer(in_dim=16, out_dim=16, dropout=0.1)
    gpsconv = GPSConv(channels=16, conv=conv, heads=4, dropout=0.1)
    out = gpsconv(x, edge_index, sph, batch)
    print(f"GPSConv output shape: {out.shape}")
    print()


# ===============================
# 4. Gradformer Model
# ===============================

class Gradformer(nn.Module):
    def __init__(self, num_layers, channels, nhead, dropout, mpnn_type="GCN"):
        """
        Gradformer: A graph transformer stacking multiple GPSConv layers.

        Args:
            num_layers (int): Number of GPSConv layers.
            channels (int): Feature dimension.
            nhead (int): Number of attention heads.
            dropout (float): Dropout probability.
            mpnn_type (str): Type of message passing layer.
        """
        super(Gradformer, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = GatedGCNLayer(
                channels, channels, dropout=dropout, residual=True
            )
            gps_conv = GPSConv(channels, conv, heads=nhead, dropout=dropout)
            self.convs.append(gps_conv)
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.size(0)
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
    edge_index = torch.tensor(
        [[i for i in range(num_nodes - 1)],
         [i + 1 for i in range(num_nodes - 1)]],
        dtype=torch.long,
    )
    batch = torch.zeros(num_nodes, dtype=torch.long)
    y = torch.rand(1)
    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
    model = Gradformer(
        num_layers=2, channels=16, nhead=4, dropout=0.1, mpnn_type="GCN"
    )
    out = model(data)
    print(f"Gradformer output shape: {out.shape}")
    print()


# ===============================
# 5. Dataset and Loss Function Test
# ===============================

def test_dataset_loss_conversion():
    print("=== Testing Dataset and Loss Function Conversion ===")
    dataset = TUDataset(root=os.path.join("/tmp", "MUTAG"), name="MUTAG")
    dataset = dataset.shuffle()
    data = dataset[0]
    print("Sample data from dataset:")
    print(data)
    output = torch.rand_like(data.y, dtype=torch.float)
    loss_fn = nn.L1Loss()
    loss = loss_fn(output.float(), data.y.float())
    print(f"Computed L1 loss on dummy sample: {loss.item()}")
    print()


if __name__ == "__main__":
    test_attention_module()
    test_gatedgcn_layer()
    test_gpsconv_module()
    test_gradformer_model()
    test_dataset_loss_conversion()
    print("All tests passed!")

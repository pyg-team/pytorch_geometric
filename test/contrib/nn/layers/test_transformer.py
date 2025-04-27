import torch
import torch.nn as nn

from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.utils.mask_utils import build_key_padding


def test_encoder_layer_forward_shapes():
    layer = GraphTransformerEncoderLayer(
        hidden_dim=32, num_heads=4, dropout=0.0
    )
    batch_size = 2
    num_nodes = 10
    x = torch.randn(batch_size * num_nodes, 32)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    key_pad = build_key_padding(batch, num_heads=4)
    attn_mask = torch.ones(
        batch_size, 4, num_nodes, num_nodes, dtype=torch.bool
    )
    out = layer(x, batch, attn_mask, key_pad)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_encoder_layer_changes_values_mha():
    layer = GraphTransformerEncoderLayer(
        hidden_dim=32, num_heads=4, dropout=0.0
    )
    # Neutralize everything except attention
    layer.norm1 = nn.Identity()
    layer.norm2 = nn.Identity()
    layer.dropout_layer = nn.Identity()
    layer.ffn = nn.Identity()

    x = torch.randn(20, 32)
    batch = torch.zeros(20, dtype=torch.long)
    key_pad = build_key_padding(batch, num_heads=4)

    out = layer(x, batch, None, key_pad)
    assert not torch.allclose(out, x)

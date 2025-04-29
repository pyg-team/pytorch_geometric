import pytest
import torch
import torch.nn as nn

from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.utils.mask_utils import build_key_padding


def make_encoder_layer(hidden_dim=32, num_heads=4, dropout=0.0):
    """Factory for a GraphTransformerEncoderLayer with given configuration."""
    return GraphTransformerEncoderLayer(
        hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
    )


@pytest.fixture(params=[(32, 4, 0.0)])
def encoder_layer(request):
    """Parameterized fixture for encoder layer configurations."""
    hidden_dim, num_heads, dropout = request.param
    return make_encoder_layer(hidden_dim, num_heads, dropout)


@pytest.fixture(params=[(2, 10), (1, 20)])
def input_batch(request):
    """Parameterized fixture for (batch_size, num_nodes) pairs."""
    batch_size, num_nodes = request.param
    hidden_dim = 32
    x = torch.randn(batch_size * num_nodes, hidden_dim)
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    return x, batch


@pytest.mark.usefixtures("encoder_layer", "input_batch")
def test_encoder_layer_forward_and_attention(encoder_layer, input_batch):
    """Test that encoder layer changes values and preserves input
    shape under attention.
    """
    x, batch = input_batch
    layer = encoder_layer
    key_pad = build_key_padding(batch, num_heads=layer.num_heads)
    num_graphs = batch.max().item() + 1
    max_nodes = x.size(0) // num_graphs
    attn_mask = torch.ones(
        num_graphs, layer.num_heads, max_nodes, max_nodes, dtype=torch.bool
    )
    out = layer(x, batch, attn_mask, key_pad)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


@pytest.mark.parametrize("neutralize_norms_and_ffn", [True, False])
def test_self_attention_effect(neutralize_norms_and_ffn):
    """Test that self-attention produces changes when other sub-layers
    are neutralized.
    """
    layer = make_encoder_layer()
    if neutralize_norms_and_ffn:
        layer.norm1 = nn.Identity()
        layer.norm2 = nn.Identity()
        layer.dropout_layer = nn.Identity()
        layer.ffn = nn.Identity()
    x = torch.randn(20, 32)
    batch = torch.zeros(20, dtype=torch.long)
    key_pad = build_key_padding(batch, num_heads=layer.num_heads)
    out = layer(x, batch, None, key_pad)
    assert not torch.allclose(out, x)

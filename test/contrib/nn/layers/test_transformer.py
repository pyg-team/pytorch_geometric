import pytest
import torch
import torch.nn as nn

from torch_geometric.contrib.utils.mask_utils import build_key_padding


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
def test_self_attention_effect(neutralize_norms_and_ffn, encoder_layer):
    """Test that self-attention produces changes when other sub-layers
    are neutralized.
    """
    layer = encoder_layer
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

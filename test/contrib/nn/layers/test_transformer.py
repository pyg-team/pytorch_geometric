import pytest
import torch
import torch.nn as nn


@pytest.mark.usefixtures("encoder_layer")
def test_encoder_layer_forward_and_attention(encoder_layer, dense_input,
                                             key_pad_mask):
    """Test that encoder layer changes values and preserves input
    shape under attention.
    """
    layer = encoder_layer
    x_dense = dense_input
    key_pad = key_pad_mask(layer.num_heads)

    B, L, _ = x_dense.shape
    attn_mask = torch.ones(B, layer.num_heads, L, L, dtype=torch.bool)

    out = layer(x_dense, struct_mask=attn_mask, key_pad=key_pad)
    assert out.shape == x_dense.shape
    assert not torch.allclose(out, x_dense)


@pytest.mark.parametrize("neutralize_norms_and_ffn", [True, False])
def test_self_attention_effect(neutralize_norms_and_ffn, encoder_layer,
                               dense_input, key_pad_mask):
    """Test that self-attention produces changes when other sub-layers
    are neutralized.
    """
    layer = encoder_layer
    if neutralize_norms_and_ffn:
        layer.norm1 = nn.Identity()
        layer.norm2 = nn.Identity()
        layer.dropout_layer = nn.Identity()
        layer.ffn = nn.Identity()

    x_dense = dense_input
    key_pad = key_pad_mask(layer.num_heads)
    out = layer(x_dense, struct_mask=None, key_pad=key_pad)
    assert not torch.allclose(out, x_dense)

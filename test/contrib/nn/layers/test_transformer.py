import pytest
import torch

from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)


def test_encoder_layer_forward_shapes():
    layer = GraphTransformerEncoderLayer(
        hidden_dim=32, num_heads=4, dropout=0.0
    )
    batch_size = 2
    num_nodes = 10
    x = torch.randn(batch_size * num_nodes, 32)
    attn_mask = torch.ones(batch_size * num_nodes, batch_size * num_nodes)
    out = layer(x, attn_mask=attn_mask)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)
    try:
        layer(x, attn_mask=attn_mask)
    except Exception as e:
        pytest.fail(
            f"Layer should accept attention mask of shape {attn_mask.shape},\
            but got error: {str(e)}"
        )


def test_encoder_layer_changes_values_mha():
    layer = GraphTransformerEncoderLayer(
        hidden_dim=32, num_heads=4, dropout=0.0
    )
    import torch.nn as nn

    # --- neutralise everything except attention --------------------------
    layer.norm1 = nn.Identity()
    layer.norm2 = nn.Identity()
    layer.dropout_layer = nn.Identity()
    layer.ffn = nn.Identity()

    x = torch.randn(20, 32)
    out = layer(x)

    # with real attention in place, the layer must alter the tensor
    assert not torch.allclose(
        out, x
    ), "Encoder alyer with MultiheadAttention should change the input tensor"

import pytest
import torch

from torch_geometric.contrib.nn.bias.spatial import GraphAttnSpatialBias
from torch_geometric.contrib.nn.models import GraphTransformer


def test_spatial_bias_shape_dtype(spatial_batch):
    bias = GraphAttnSpatialBias(num_heads=4, num_spatial=10)
    batch = spatial_batch(2, 5, 10)
    out = bias(batch)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (2, 4, 5, 5)


def test_spatial_bias_transformer_affects(spatial_batch):
    torch.manual_seed(12345)
    provider = GraphAttnSpatialBias(num_heads=4, num_spatial=8)
    batch = spatial_batch(1, 6, 8, feat_dim=16)
    encoder_cfg_with_bias = {
        'attn_bias_providers': [provider],
        'num_encoder_layers': 1
    }
    encoder_cfg_without_bias = {'num_encoder_layers': 1}
    m0 = GraphTransformer(hidden_dim=16, out_channels=3,
                          encoder_cfg=encoder_cfg_without_bias).eval()
    m1 = GraphTransformer(hidden_dim=16, out_channels=3,
                          encoder_cfg=encoder_cfg_with_bias).eval()

    with torch.no_grad():
        o0 = m0(batch)
        o1 = m1(batch)

    assert o0.shape == o1.shape
    assert not torch.allclose(o0, o1, rtol=1e-4, atol=1e-4)


def test_spatial_bias_gradients(spatial_batch):
    provider = GraphAttnSpatialBias(num_heads=4, num_spatial=8)
    encoder_cfg = {'attn_bias_providers': [provider], 'num_encoder_layers': 1}
    model = GraphTransformer(hidden_dim=16, out_channels=3,
                             encoder_cfg=encoder_cfg)
    batch = spatial_batch(1, 5, 8, feat_dim=16)
    out = model(batch)
    out.sum().backward()

    for name, param in provider.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert not torch.all(param.grad == 0), f"{name} grad is zero"


@pytest.mark.parametrize(
    "num_heads,num_spatial,use_super",
    [
        (1, 5, False),
        (4, 5, False),
        (1, 5, True),
        (4, 10, True),
    ],
)
def test_spatial_bias_config(spatial_batch, num_heads, num_spatial, use_super):
    bias = GraphAttnSpatialBias(num_heads=num_heads, num_spatial=num_spatial,
                                use_super_node=use_super)
    batch = spatial_batch(3, 7, num_spatial)
    out = bias(batch)

    exp_len = 7 + (1 if use_super else 0)
    assert out.shape == (3, num_heads, exp_len, exp_len)

    if use_super:
        r0 = out[0, 0, 0]
        r1 = out[0, 0, 1]
        assert not torch.allclose(r0, r1)


def test_spatial_bias_supports_variable_graph_sizes(hetero_spatial_batch):
    """Test that the spatial bias provider can handle variable graph sizes."""
    data = hetero_spatial_batch([11, 14], feat_dim=8, num_spatial=5)
    provider = GraphAttnSpatialBias(num_heads=4, num_spatial=5)
    _ = provider(data)


def test_spatial_bias_handles_heterogeneous_blockdiag_spatial_pos(
        blockdiag_spatial_batch):
    """GraphAttnSpatialBias must pad per-graph masks before stacking."""
    provider = GraphAttnSpatialBias(num_heads=4, num_spatial=5)
    _ = provider(blockdiag_spatial_batch)

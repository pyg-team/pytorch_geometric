# test/contrib/nn/bias/test_spatial_bias.py

import pytest
import torch

from torch_geometric.contrib.nn.bias.spatial import GraphAttnSpatialBias
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


def make_batch_with_spatial(
    batch_size: int,
    seq_len: int,
    num_spatial: int,
    feature_dim: int = 1,
) -> Batch:
    """Create a Batch of graphs each carrying spatial_pos and x."""
    data_list = []
    for _ in range(batch_size):
        # spatial distances
        spatial_pos = torch.randint(0, num_spatial, (seq_len, seq_len))
        # node features of the requested dimension
        x = torch.randn(seq_len, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(
            Data(
                x=x,
                spatial_pos=spatial_pos,
                edge_index=edge_index,
            )
        )
    return Batch.from_data_list(data_list)


def test_spatial_bias_shape():
    """Test that forward pass returns correct tensor shape and dtype."""
    num_heads = 4
    num_spatial = 10
    seq_len = 5
    batch_size = 2

    bias_provider = GraphAttnSpatialBias(
        num_heads=num_heads,
        num_spatial=num_spatial,
    )
    # feature_dim=1 is fine here; we only call bias_provider.forward
    batch = make_batch_with_spatial(
        batch_size, seq_len, num_spatial, feature_dim=1
    )

    bias = bias_provider(batch)
    assert isinstance(bias, torch.Tensor)
    assert bias.dtype == torch.float32, "Expected float32 output"
    assert bias.shape == (batch_size, num_heads, seq_len,
                          seq_len), (f"Wrong shape: {bias.shape}")


def test_spatial_bias_affects_transformer():
    """Test that using spatial bias provider changes transformer outputs."""
    torch.manual_seed(12345)
    num_heads = 4
    num_spatial = 8
    seq_len = 6
    feature_dim = 16  # must match GraphTransformer.hidden_dim

    bias_provider = GraphAttnSpatialBias(
        num_heads=num_heads,
        num_spatial=num_spatial,
    )
    batch = make_batch_with_spatial(
        1, seq_len, num_spatial, feature_dim=feature_dim
    )

    # Ensure batch.x has the right feature dimension for the model
    # (Batch.from_data_list already set this correctly via helper)

    model_no = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=3,
        num_encoder_layers=1,
    )
    model_yes = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=3,
        num_encoder_layers=1,
        attn_bias_providers=[bias_provider],
    )

    model_no.eval()
    model_yes.eval()
    with torch.no_grad():
        out1 = model_no(batch)["logits"]
        out2 = model_yes(batch)["logits"]

    assert out1.shape == out2.shape
    assert not torch.allclose(
        out1, out2, rtol=1e-4, atol=1e-4
    ), ("Outputs should differ when spatial bias is applied")


def test_spatial_bias_gradients():
    """Test that gradients flow through spatial bias provider parameters."""
    num_heads = 4
    num_spatial = 8
    seq_len = 5
    feature_dim = 16

    bias_provider = GraphAttnSpatialBias(
        num_heads=num_heads,
        num_spatial=num_spatial,
    )
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=3,
        num_encoder_layers=1,
        attn_bias_providers=[bias_provider],
    )
    batch = make_batch_with_spatial(
        1, seq_len, num_spatial, feature_dim=feature_dim
    )

    out = model(batch)["logits"]
    loss = out.sum()
    loss.backward()

    for name, param in bias_provider.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no grad"
        assert not torch.all(param.grad == 0), \
            f"Parameter {name} has zero gradient"


@pytest.mark.parametrize(
    "num_heads,num_spatial,use_super_node",
    [
        (1, 5, False),
        (4, 5, False),
        (1, 5, True),
        (4, 10, True),
    ],
)
def test_spatial_bias_config(num_heads, num_spatial, use_super_node):
    """Test GraphAttnSpatialBias with various configurations."""
    seq_len = 7
    batch_size = 3

    bias_provider = GraphAttnSpatialBias(
        num_heads=num_heads,
        num_spatial=num_spatial,
        use_super_node=use_super_node,
    )
    # feature_dim=1 is sufficient for provider-only tests
    batch = make_batch_with_spatial(
        batch_size, seq_len, num_spatial, feature_dim=1
    )

    bias = bias_provider(batch)
    expected_seq_len = seq_len + (1 if use_super_node else 0)
    assert bias.shape == (
        batch_size, num_heads, expected_seq_len, expected_seq_len
    )
    assert bias.dtype == torch.float32

    if use_super_node:
        # check that row 0 differs from row 1 for graph 0, head 0
        row0 = bias[0, 0, 0, :]
        row1 = bias[0, 0, 1, :]
        assert not torch.allclose(row0, row1), "Super-node row should differ"

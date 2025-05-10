import pytest
import torch

from torch_geometric.contrib.nn.bias.hop import GraphAttnHopBias
from torch_geometric.contrib.nn.models import GraphTransformer


def test_hop_bias_shape_and_dtype(hop_batch):
    num_heads = 4
    num_hops = 10
    seq_len = 5
    batch_size = 2

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    batch = hop_batch(batch_size, seq_len, num_hops)
    bias = bias_provider(batch)

    assert isinstance(bias, torch.Tensor)
    assert bias.dtype == torch.float32
    assert bias.shape == (batch_size, num_heads, seq_len, seq_len)


def test_hop_bias_affects_transformer(hop_batch):
    torch.manual_seed(0)
    num_heads = 4
    num_hops = 8
    seq_len = 6
    feature_dim = 16

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    batch = hop_batch(1, seq_len, num_hops, feature_dim)

    model_plain = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=3,
        num_encoder_layers=1,
    )
    model_biased = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=3,
        num_encoder_layers=1,
        attn_bias_providers=[bias_provider],
    )

    model_plain.eval()
    model_biased.eval()
    with torch.no_grad():
        out_plain = model_plain(batch)
        out_biased = model_biased(batch)

    assert out_plain.shape == out_biased.shape
    assert not torch.allclose(out_plain, out_biased, rtol=1e-4, atol=1e-4)


def test_hop_bias_gradients(hop_batch):
    num_heads = 4
    num_hops = 8
    seq_len = 5
    feature_dim = 16

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=3,
        num_encoder_layers=1,
        attn_bias_providers=[bias_provider],
    )
    batch = hop_batch(1, seq_len, num_hops, feature_dim)

    out = model(batch)
    loss = out.sum()
    loss.backward()

    for name, param in bias_provider.named_parameters():
        assert param.grad is not None, f"Parameter {name!r} has no gradient"
        assert not torch.all(param.grad == 0
                             ), (f"Parameter {name!r} gradient is zero")


@pytest.mark.parametrize(
    "num_heads, num_hops, use_super_node",
    [
        (1, 5, False),
        (4, 5, False),
        (1, 5, True),
        (4, 10, True),
    ],
)
def test_hop_bias_configuration(
    hop_batch, num_heads, num_hops, use_super_node
):
    seq_len = 7
    batch_size = 3

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
        use_super_node=use_super_node,
    )
    batch = hop_batch(batch_size, seq_len, num_hops)
    bias = bias_provider(batch)

    expected_len = seq_len + (1 if use_super_node else 0)
    assert bias.shape == (batch_size, num_heads, expected_len, expected_len)
    assert bias.dtype == torch.float32

    if use_super_node:
        row0 = bias[0, 0, 0]
        row1 = bias[0, 0, 1]
        assert not torch.allclose(row0, row1), "Super-node row should differ"


def test_hop_bias_handles_variable_graph_sizes(blockdiag_hop_batch):
    """GraphAttnHopBias should pad ragged hop_dist blocks to (B, H, L, L)."""
    node_counts = [20, 17, 13]
    data = blockdiag_hop_batch(node_counts, feat_dim=8, num_hops=10)
    provider = GraphAttnHopBias(num_heads=4, num_hops=10)
    out = provider(data)
    B = len(node_counts)
    L = max(node_counts)
    assert out.shape == (B, 4, L, L)

import pytest
import torch

from torch_geometric.contrib.nn.bias.hop import GraphAttnHopBias
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


@pytest.fixture
def make_hop_batch():
    """Factory for creating Batch objects with hop_dist and x."""

    def _make(batch_size, seq_len, num_hops, feature_dim=1):
        data_list = []
        for _ in range(batch_size):
            hop_dist = torch.randint(0, num_hops, (seq_len, seq_len))
            x = torch.randn(seq_len, feature_dim)
            data = Data(
                x=x,
                hop_dist=hop_dist,
                edge_index=torch.empty((2, 0), dtype=torch.long),
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    return _make


def test_hop_bias_shape_and_dtype(make_hop_batch):
    num_heads = 4
    num_hops = 10
    seq_len = 5
    batch_size = 2

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    batch = make_hop_batch(batch_size, seq_len, num_hops)
    bias = bias_provider(batch)

    assert isinstance(bias, torch.Tensor)
    assert bias.dtype == torch.float32
    assert bias.shape == (batch_size, num_heads, seq_len, seq_len)


def test_hop_bias_affects_transformer(make_hop_batch):
    torch.manual_seed(0)
    num_heads = 4
    num_hops = 8
    seq_len = 6
    feature_dim = 16

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    batch = make_hop_batch(1, seq_len, num_hops, feature_dim)

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
        out_plain = model_plain(batch)["logits"]
        out_biased = model_biased(batch)["logits"]

    assert out_plain.shape == out_biased.shape
    assert not torch.allclose(out_plain, out_biased, rtol=1e-4, atol=1e-4)


def test_hop_bias_gradients(make_hop_batch):
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
    batch = make_hop_batch(1, seq_len, num_hops, feature_dim)

    out = model(batch)["logits"]
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
    make_hop_batch, num_heads, num_hops, use_super_node
):
    seq_len = 7
    batch_size = 3

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
        use_super_node=use_super_node,
    )
    batch = make_hop_batch(batch_size, seq_len, num_hops)
    bias = bias_provider(batch)

    expected_len = seq_len + (1 if use_super_node else 0)
    assert bias.shape == (batch_size, num_heads, expected_len, expected_len)
    assert bias.dtype == torch.float32

    if use_super_node:
        row0 = bias[0, 0, 0]
        row1 = bias[0, 0, 1]
        assert not torch.allclose(row0, row1), "Super-node row should differ"

import pytest
import torch

from torch_geometric.contrib.nn.bias.hop import GraphAttnHopBias
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


def make_batch_with_hop(
    batch_size: int,
    seq_len: int,
    num_hops: int,
    feature_dim: int = 1,
) -> Batch:
    """Create a Batch of graphs each carrying hop_dist and x."""
    data_list = []
    for _ in range(batch_size):
        hop_dist = torch.randint(0, num_hops, (seq_len, seq_len))
        x = torch.randn(seq_len, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(
            Data(
                x=x,
                hop_dist=hop_dist,
                edge_index=edge_index,
            )
        )
    return Batch.from_data_list(data_list)


def test_hop_bias_shape():
    num_heads = 4
    num_hops = 10
    seq_len = 5
    batch_size = 2

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    batch = make_batch_with_hop(batch_size, seq_len, num_hops, feature_dim=1)
    bias = bias_provider(batch)

    assert isinstance(bias, torch.Tensor)
    assert bias.dtype == torch.float32
    assert bias.shape == (batch_size, num_heads, seq_len, seq_len)


def test_hop_bias_affects_transformer():
    torch.manual_seed(0)
    num_heads = 4
    num_hops = 8
    seq_len = 6
    feature_dim = 16

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
    )
    batch = make_batch_with_hop(1, seq_len, num_hops, feature_dim=feature_dim)

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
    assert not torch.allclose(out1, out2, rtol=1e-4, atol=1e-4)


def test_hop_bias_gradients():
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
    batch = make_batch_with_hop(1, seq_len, num_hops, feature_dim=feature_dim)

    out = model(batch)["logits"]
    loss = out.sum()
    loss.backward()

    for name, param in bias_provider.named_parameters():
        assert param.grad is not None, f"Param {name} has no grad"
        assert not torch.all(param.grad == 0), f"Param {name} grad is zero"


@pytest.mark.parametrize(
    "num_heads,num_hops,use_super_node",
    [
        (1, 5, False),
        (4, 5, False),
        (1, 5, True),
        (4, 10, True),
    ],
)
def test_hop_bias_config(num_heads, num_hops, use_super_node):
    seq_len = 7
    batch_size = 3

    bias_provider = GraphAttnHopBias(
        num_heads=num_heads,
        num_hops=num_hops,
        use_super_node=use_super_node,
    )
    batch = make_batch_with_hop(batch_size, seq_len, num_hops, feature_dim=1)
    bias = bias_provider(batch)
    expected_seq_len = seq_len + (1 if use_super_node else 0)

    assert bias.shape == (
        batch_size, num_heads, expected_seq_len, expected_seq_len
    )
    assert bias.dtype == torch.float32

    if use_super_node:
        row0 = bias[0, 0, 0, :]
        row1 = bias[0, 0, 1, :]
        assert not torch.allclose(row0, row1)

import pytest
import torch

from torch_geometric.contrib.nn.bias.edge import GraphAttnEdgeBias
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch


@pytest.mark.parametrize(
    "num_heads,num_edges,seq_len,batch_size", [
        (4, 12, 5, 3),
    ]
)
def test_edge_bias_shape_dtype(
    edge_batch, num_heads, num_edges, seq_len, batch_size
):
    bias = GraphAttnEdgeBias(num_heads=num_heads, num_edges=num_edges)
    batch = edge_batch(batch_size, seq_len, num_edges)
    out = bias(batch)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (batch_size, num_heads, seq_len, seq_len)


def test_edge_bias_transformer_affects(edge_batch):
    torch.manual_seed(0)
    provider = GraphAttnEdgeBias(num_heads=4, num_edges=8)
    batch = edge_batch(1, 6, 8, feat_dim=16)

    m0 = GraphTransformer(
        hidden_dim=16, num_class=2, num_encoder_layers=1
    ).eval()
    m1 = GraphTransformer(
        hidden_dim=16,
        num_class=2,
        num_encoder_layers=1,
        attn_bias_providers=[provider]
    ).eval()

    with torch.no_grad():
        out0 = m0(batch)
        out1 = m1(batch)

    assert out0.shape == out1.shape
    assert not torch.allclose(out0, out1, rtol=1e-4, atol=1e-4)


def test_edge_bias_gradients(edge_batch):
    provider = GraphAttnEdgeBias(num_heads=4, num_edges=7)
    model = GraphTransformer(
        hidden_dim=16,
        num_class=2,
        num_encoder_layers=1,
        attn_bias_providers=[provider]
    )
    batch = edge_batch(1, 5, 7, feat_dim=16)

    out = model(batch)
    out.sum().backward()

    for name, param in provider.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert not torch.all(param.grad == 0), f"{name} grad is zero"


@pytest.mark.parametrize(
    "num_heads,num_edges,edge_type,mhmd,use_super",
    [
        (1, 5, 'simple', None, False),
        (2, 6, 'simple', None, True),
        (1, 5, 'multi_hop', 3, False),
        (3, 9, 'multi_hop', 5, True),
    ],
)
def test_edge_bias_config(
    edge_batch, num_heads, num_edges, edge_type, mhmd, use_super
):
    bias = GraphAttnEdgeBias(
        num_heads=num_heads,
        num_edges=num_edges,
        edge_type=edge_type,
        multi_hop_max_dist=mhmd,
        use_super_node=use_super
    )
    batch = edge_batch(2, 7, num_edges)
    out = bias(batch)

    exp_len = 7 + (1 if use_super else 0)
    assert out.shape == (2, num_heads, exp_len, exp_len)

    if use_super:
        r0 = out[0, 0, 0]
        r1 = out[0, 0, 1]
        assert not torch.allclose(r0, r1)


def test_edge_bias_uniform_for_zero_dist(simple_batch):
    """Missing edge_dist → all entries use the same learned embedding."""
    # Build a batch of two identical 4-node graphs
    g = simple_batch(8, 4)
    batch = Batch.from_data_list(g.to_data_list() + g.to_data_list())

    provider = GraphAttnEdgeBias(num_heads=3, num_edges=5)
    out = provider(batch)

    # Expect shape (2 graphs, 3 heads, 4 nodes, 4 nodes)
    assert out.shape == (2, 3, 4, 4)
    assert out.dtype == torch.float32

    # Since every distance is effectively zero, the bias for each head/graph
    # should be constant across all node pairs.
    # Compare all entries to the (0,0) position of each head/graph.
    constant = out[:, :, 0, 0].view(2, 3, 1, 1)
    assert torch.allclose(out, constant.expand_as(out))


def test_edge_bias_changes_transformer_without_dist(simple_batch):
    """Even without explicit edge_dist, the learned bias alters
    transformer outputs.
    """
    g = simple_batch(8, 5)
    N_heads = 4
    batch = Batch.from_data_list(g.to_data_list() + g.to_data_list())

    provider = GraphAttnEdgeBias(num_heads=N_heads, num_edges=7)

    base = GraphTransformer(
        hidden_dim=8,
        num_class=2,
        num_encoder_layers=1,
    ).eval()
    biased = GraphTransformer(
        hidden_dim=8,
        num_class=2,
        num_encoder_layers=2,
        attn_bias_providers=[provider],
    ).eval()

    with torch.no_grad():
        out_base = base(batch)
        out_biased = biased(batch)

    # Shapes must agree...
    assert out_base.shape == out_biased.shape
    # ...but learned edge bias (for distance=0) is nonzero, so outputs differ
    assert not torch.allclose(out_base, out_biased)


@pytest.mark.parametrize(
    "edge_type, multi_hop_max_dist, distances_input", [
        ("simple", None, torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)),
        ("multi_hop", 2, torch.tensor([[[0, 3], [2, 4]]], dtype=torch.long))
    ]
)
def test_embed_bias_via_param(edge_type, multi_hop_max_dist, distances_input):
    bias_provider = GraphAttnEdgeBias(
        num_heads=2,
        num_edges=4,
        edge_type=edge_type,
        multi_hop_max_dist=multi_hop_max_dist
    )
    result = bias_provider._embed_bias(distances_input)
    if edge_type == "multi_hop" and multi_hop_max_dist is not None:
        distances_expected = distances_input.clamp(max=multi_hop_max_dist)
    else:
        distances_expected = distances_input
    expected = bias_provider.edge_embeddings(distances_expected)
    assert torch.equal(result, expected), \
        "Embedding bias does not match expected output"


@pytest.mark.parametrize("use_super", [False, True])
def test_edge_bias_handles_variable_graph_sizes(
    blockdiag_edge_batch, use_super
):
    """GraphAttnEdgeBias should pad ragged edge_dist blocks to (B, H, L, L)."""
    node_counts = [12, 7, 19]
    data = blockdiag_edge_batch(node_counts, feat_dim=8, num_edges=5)
    provider = GraphAttnEdgeBias(
        num_heads=3, num_edges=5, use_super_node=use_super
    )
    out = provider(data)
    B = len(node_counts)
    L = max(node_counts) + (1 if use_super else 0)
    assert out.shape == (B, 3, L, L)

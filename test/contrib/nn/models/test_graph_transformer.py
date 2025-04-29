# test/contrib/nn/models/test_graph_transformer_refactored.py

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.contrib.nn.bias import (
    GraphAttnEdgeBias,
    GraphAttnSpatialBias,
)
from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# ── Fixtures & Factories ─────────────────────────────────────────────────────


@pytest.fixture
def simple_batch():
    """Factory for a single-graph Batch with fully-connected edges."""

    def _make(feat_dim, num_nodes):
        x = torch.randn(num_nodes, feat_dim)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)])

    return _make


@pytest.fixture
def spatial_batch():
    """Factory for a batch with spatial_pos but no edges."""

    def _make(batch_size, seq_len, num_spatial, feature_dim):
        data_list = []
        for _ in range(batch_size):
            spatial_pos = torch.randint(0, num_spatial, (seq_len, seq_len))
            x = torch.randn(seq_len, feature_dim)
            data_list.append(
                Data(
                    x=x,
                    spatial_pos=spatial_pos,
                    edge_index=torch.empty((2, 0), dtype=torch.long)
                )
            )
        return Batch.from_data_list(data_list)

    return _make


@pytest.fixture
def spatial_edge_batch():
    """Factory for a batch with spatial_pos and edge_dist."""

    def _make(batch_size, seq_len, num_spatial, num_edges, feature_dim):
        data_list = []
        for _ in range(batch_size):
            spatial_pos = torch.randint(0, num_spatial, (seq_len, seq_len))
            edge_dist = torch.randint(0, num_edges, (seq_len, seq_len))
            x = torch.randn(seq_len, feature_dim)
            data_list.append(
                Data(
                    x=x,
                    spatial_pos=spatial_pos,
                    edge_dist=edge_dist,
                    edge_index=torch.empty((2, 0), dtype=torch.long)
                )
            )
        return Batch.from_data_list(data_list)

    return _make


class AddOneLayer(nn.Module):

    def __init__(self, num_heads=1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, *args, **kwargs):
        return x + 1.0


class ConstantDegreeEncoder(nn.Module):

    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def forward(self, data: Data) -> torch.Tensor:
        return torch.full_like(data.x, self.value)


class IdentityEncoder(nn.Module):

    def forward(self, x, batch=None, attn_mask=None):
        return x


class TraceableWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, batch: Tensor, bias: Tensor = None):
        data = Batch(
            x=x,
            batch=batch,
            edge_index=torch.empty((2, 0), dtype=torch.long, device=x.device)
        )
        if bias is not None:
            data.bias = bias
        return self.model(data)["logits"]


# ── Core Forward‐Shape Tests ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "num_graphs,num_nodes,feat_dim,num_classes", [
        (1, 10, 16, 2),
        (5, 20, 32, 3),
        (3, 15, 64, 4),
    ],
    ids=["single", "multi_diff", "multi_same"]
)
def test_forward_shape(
    simple_batch, num_graphs, num_nodes, feat_dim, num_classes
):
    batches = [
        simple_batch(feat_dim, num_nodes).to_data_list()[0]
        for _ in range(num_graphs)
    ]
    batch = Batch.from_data_list(batches)
    model = GraphTransformer(hidden_dim=feat_dim, num_class=num_classes)
    out = model(batch)["logits"]
    assert out.shape == (num_graphs, num_classes)


def test_forward_without_gnn_hook(simple_batch):
    batch = simple_batch(feat_dim=8, num_nodes=2)
    model = GraphTransformer(hidden_dim=8)
    out = model(batch)["logits"]
    assert out.shape == (1, model.classifier.out_features)


# ── Feature & Structural Encoder Tests ───────────────────────────────────────


def test_super_node_readout(simple_batch):
    data = simple_batch(feat_dim=4, num_nodes=1)
    model = GraphTransformer(hidden_dim=4, num_class=2, use_super_node=True)
    model.encoder = IdentityEncoder()  # bypass transformer
    out = model(data)["logits"]
    expected = model.classifier(model.cls_token.expand(1, -1))
    assert torch.allclose(out, expected)


def test_node_feature_encoder_identity(simple_batch):
    batch = simple_batch(feat_dim=16, num_nodes=10)
    model = GraphTransformer(
        hidden_dim=16, num_class=2, node_feature_encoder=nn.Identity()
    )
    out1 = model(batch)["logits"]
    batch.x = 2 * batch.x
    out2 = model(batch)["logits"]
    assert not torch.allclose(out1, out2)


# ── Transformer Core Tests ──────────────────────────────────────────────────


def test_transformer_block_identity(simple_batch):
    batch = simple_batch(feat_dim=16, num_nodes=8)
    model = GraphTransformer(hidden_dim=16, num_class=4, num_encoder_layers=1)
    model.encoder[0] = AddOneLayer()
    out = model(batch)["logits"]
    assert out.shape == (1, 4)


def test_encoder_layer_type():
    model = GraphTransformer(hidden_dim=16, num_encoder_layers=1)
    assert isinstance(model.encoder[0], GraphTransformerEncoderLayer)


def test_torchscript_trace():
    G, N, F, C = 2, 5, 8, 3
    x = torch.randn(G * N, F)
    batch = torch.arange(G).repeat_interleave(N)
    bias = torch.rand(G, 4, N, N)
    model = GraphTransformer(
        hidden_dim=F, num_class=C, num_encoder_layers=2
    ).eval()
    wrapper = TraceableWrapper(model)
    traced = torch.jit.trace(wrapper, (x, batch, bias))
    assert torch.allclose(
        wrapper(x, batch, bias), traced(x, batch, bias), atol=1e-6
    )


def test_backward(simple_batch):
    batch = simple_batch(feat_dim=8, num_nodes=5)
    model = GraphTransformer(
        hidden_dim=8, num_class=3, use_super_node=True, num_encoder_layers=1
    )
    model.encoder[0] = AddOneLayer()
    out = model(batch)["logits"]
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None


# ── Attention & Bias Tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("num_heads", [1, 4])
def test_attention_mask_changes_logits(simple_batch, num_heads):
    batch = simple_batch(feat_dim=8, num_nodes=5)
    model = GraphTransformer(hidden_dim=8, num_class=3, num_encoder_layers=1)
    model.encoder[0] = GraphTransformerEncoderLayer(
        hidden_dim=8, num_heads=num_heads
    )
    out1 = model(batch)["logits"]
    batch.bias = torch.randint(0, 2, (1, 5, 5), dtype=torch.bool)
    out2 = model(batch)["logits"]
    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2)


def test_cls_token_transformation(simple_batch):
    batch = simple_batch(feat_dim=8, num_nodes=3)
    model = GraphTransformer(
        hidden_dim=8, num_class=2, use_super_node=True, num_encoder_layers=1
    )
    orig = model.cls_token.clone()
    x, bv = model._prepend_cls_token_flat(batch.x, batch.batch)
    enc = model.encoder(x, bv)
    # first positions are the CLS tokens
    sizes = torch.bincount(bv)
    idx = torch.cumsum(sizes, 0) - sizes
    transformed = enc[idx]
    assert not torch.allclose(transformed, orig.expand(2, -1))


@pytest.mark.parametrize("seq_len,num_spatial,feat_dim", [(10, 4, 16)])
def test_spatial_bias_shifts_logits(
    spatial_batch, seq_len, num_spatial, feat_dim
):
    batch0 = spatial_batch(1, seq_len, num_spatial, feat_dim)
    batch1 = spatial_batch(1, seq_len, num_spatial, feat_dim)
    bias = torch.zeros((1, num_spatial, seq_len, seq_len))
    bias[:, :, 0, 1] = 5.0
    batch1.bias = bias
    model = GraphTransformer(
        hidden_dim=feat_dim, num_class=2, num_encoder_layers=1
    )
    out0 = model(batch0)["logits"]
    out1 = model(batch1)["logits"]
    assert not torch.allclose(out0, out1)


@pytest.mark.parametrize("use_super_node", [False])
def test_multi_provider_fusion(spatial_edge_batch, use_super_node):
    batch = spatial_edge_batch(
        2, seq_len=4, num_spatial=5, num_edges=6, feature_dim=8
    )
    provs = [
        GraphAttnSpatialBias(2, 5, use_super_node),
        GraphAttnEdgeBias(2, 6, use_super_node)
    ]
    model = GraphTransformer(
        hidden_dim=8,
        num_class=2,
        use_super_node=use_super_node,
        attn_bias_providers=provs
    )
    fused = model._collect_attn_bias(batch)
    direct = sum(p(batch).to(fused.dtype).to(fused.device) for p in provs)
    assert fused.shape == direct.shape
    assert torch.allclose(fused, direct)


# ── GNN‐Hook Tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("pos", ["pre", "post", "parallel"])
def test_gnn_hook_order(simple_batch, pos):
    seq = []

    def gnn(data, x):
        seq.append("gnn")
        return x

    model = GraphTransformer(
        hidden_dim=8,
        num_class=2,
        num_encoder_layers=1,
        gnn_block=gnn,
        gnn_position=pos
    )
    # patch the real encoder
    orig = model.encoder.layers[0].forward

    def track(x, b, mask, kp):
        seq.append("enc")
        return orig(x, b, mask, kp)

    model.encoder.layers[0].forward = track

    _ = model(simple_batch(8, 3))
    if pos == "pre":
        assert seq == ["gnn", "enc"]
    elif pos == "post":
        assert seq == ["enc", "gnn"]
    else:
        assert sorted(seq) == ["enc", "gnn"]


# ── GNN‐Block Parametrized Tests ────────────────────────────────────────────


@pytest.mark.parametrize(
    "conv,where", [(GCNConv, "pre"), (SAGEConv, "post"), (GATConv, "pre")]
)
def test_gnn_block_real_convs(simple_batch, conv, where):
    base_out = GraphTransformer(
        hidden_dim=16, num_class=4, num_encoder_layers=2
    )(simple_batch(16, 5))["logits"]

    def gnn(data, x):
        return conv(x.size(-1), x.size(-1))(x, data.edge_index)

    out = GraphTransformer(
        hidden_dim=16,
        num_class=4,
        num_encoder_layers=2,
        gnn_block=gnn,
        gnn_position=where
    )(simple_batch(16, 5))["logits"]
    assert out.shape == base_out.shape
    assert not torch.allclose(out, base_out)


@pytest.mark.parametrize("where", ["pre", "post", "parallel"])
def test_gnn_block_mlp(simple_batch, where):
    base_out = GraphTransformer(
        hidden_dim=8, num_class=3, num_encoder_layers=1
    )(simple_batch(8, 4))["logits"]

    def mlp(data, x):
        y = nn.Linear(x.size(-1), x.size(-1))(x)
        return nn.ReLU()(nn.Linear(x.size(-1), x.size(-1))(y))

    out = GraphTransformer(
        hidden_dim=8,
        num_class=3,
        num_encoder_layers=1,
        gnn_block=mlp,
        gnn_position=where
    )(simple_batch(8, 4))["logits"]
    assert out.shape == base_out.shape
    assert not torch.allclose(out, base_out)

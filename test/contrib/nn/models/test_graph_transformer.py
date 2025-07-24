import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Use alias imports to avoid yapf/isort conflicts with long module names
import torch_geometric.contrib.nn.bias as _bias
import torch_geometric.contrib.nn.layers.transformer as _transformer
import torch_geometric.contrib.nn.positional as _positional
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.models.graph_transformer import (
    DEFAULT_ENCODER,
    DEFAULT_GNN,
)
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# Extract classes from alias imports
GraphAttnEdgeBias = _bias.GraphAttnEdgeBias
GraphAttnSpatialBias = _bias.GraphAttnSpatialBias
GraphTransformerEncoderLayer = _transformer.GraphTransformerEncoderLayer
DegreeEncoder = _positional.DegreeEncoder
EigEncoder = _positional.EigEncoder
SVDEncoder = _positional.SVDEncoder

CUDA = torch.cuda.is_available()


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
    def __init__(self, num_heads=1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, *args, **kwargs):
        return x


class TraceableWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, batch: Tensor, bias: Tensor = None):
        data = Batch(
            x=x, batch=batch, edge_index=torch.empty((2, 0), dtype=torch.long,
                                                     device=x.device))
        if bias is not None:
            data.bias = bias
        return self.model(data)


# ── Core Forward‐Shape Tests ─────────────────────────────────────────────────


@pytest.mark.parametrize("num_graphs,num_nodes,feat_dim,out_channels", [
    (1, 10, 16, 2),
    (5, 20, 32, 3),
    (3, 15, 64, 4),
], ids=["single", "multi_diff", "multi_same"])
def test_forward_shape(simple_batch, num_graphs, num_nodes, feat_dim,
                       out_channels):
    batches = [
        simple_batch(feat_dim, num_nodes).to_data_list()[0]
        for _ in range(num_graphs)
    ]
    batch = Batch.from_data_list(batches)
    encoder_cfg = {}
    model = GraphTransformer(hidden_dim=feat_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    out = model(batch)
    total_nodes = num_graphs * num_nodes
    assert out.shape == (total_nodes, out_channels)


def test_forward_without_gnn_hook(simple_batch):
    num_nodes = 5
    feat_dim = 8
    hidden_dim = 8  # keep in-dim == hidden_dim when using Identity
    out_channels = 3
    batch = simple_batch(feat_dim=feat_dim, num_nodes=num_nodes)
    encoder_cfg = {}
    model = GraphTransformer(hidden_dim=hidden_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    out = model(batch)
    assert out.shape == (num_nodes, out_channels)


# ── Feature & Structural Encoder Tests ───────────────────────────────────────


def test_super_node_readout(simple_batch):
    data = simple_batch(feat_dim=4, num_nodes=1)
    NUM_HEADS = 4
    encoder_cfg = {
        **DEFAULT_ENCODER, "num_heads": NUM_HEADS,
        "use_super_node": True
    }
    model = GraphTransformer(hidden_dim=4, out_channels=2,
                             encoder_cfg=encoder_cfg)
    model.encoder = IdentityEncoder(num_heads=NUM_HEADS)
    out = model(data)
    expected = model.output_projection(model.cls_token)
    assert torch.allclose(out[0:1], expected)
    assert out.shape[0] == data.num_nodes + 1


def test_node_feature_encoder_identity(simple_batch):
    batch = simple_batch(feat_dim=16, num_nodes=10)
    encoder_cfg = {**DEFAULT_ENCODER, "node_feature_encoder": nn.Identity()}
    model = GraphTransformer(hidden_dim=16, out_channels=2,
                             encoder_cfg=encoder_cfg)
    out1 = model(batch)
    batch.x = 2 * batch.x
    out2 = model(batch)
    assert not torch.allclose(out1, out2)


# ── Transformer Core Tests ──────────────────────────────────────────────────


def test_transformer_block_identity(simple_batch):
    num_nodes = 5
    feat_dim = 8
    hidden_dim = 16
    out_channels = 3
    batch = simple_batch(feat_dim=feat_dim, num_nodes=num_nodes)
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 1}
    model = GraphTransformer(hidden_dim=hidden_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    model.encoder[0] = AddOneLayer()
    out = model(batch)
    assert out.shape == (num_nodes, out_channels)


def test_encoder_layer_type():
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 1}
    model = GraphTransformer(hidden_dim=16, out_channels=1,
                             encoder_cfg=encoder_cfg)
    assert isinstance(model.encoder[0], GraphTransformerEncoderLayer)


def test_torchscript_trace():
    G, N, F, C = 2, 5, 8, 3
    x = torch.randn(G * N, F)
    batch = torch.arange(G).repeat_interleave(N)
    bias = torch.rand(G, 4, N, N)
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 2}
    model = GraphTransformer(hidden_dim=F, out_channels=C,
                             encoder_cfg=encoder_cfg).eval()
    wrapper = TraceableWrapper(model)
    traced = torch.jit.trace(wrapper, (x, batch, bias))
    assert torch.allclose(wrapper(x, batch, bias), traced(x, batch, bias),
                          atol=1e-6)


def test_backward(simple_batch):
    batch = simple_batch(feat_dim=8, num_nodes=5)
    encoder_cfg = {
        **DEFAULT_ENCODER, "num_encoder_layers": 1,
        "use_super_node": True
    }
    model = GraphTransformer(hidden_dim=8, out_channels=3,
                             encoder_cfg=encoder_cfg)
    model.encoder[0] = AddOneLayer()
    out = model(batch)
    loss = out.sum()
    loss.backward()
    for _, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None


# ── Attention & Bias Tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("num_heads", [1, 4])
def test_attention_mask_changes_logits(simple_batch, num_heads):
    batch = simple_batch(feat_dim=8, num_nodes=5)
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 1}
    model = GraphTransformer(hidden_dim=8, out_channels=3,
                             encoder_cfg=encoder_cfg)
    model.encoder[0] = GraphTransformerEncoderLayer(hidden_dim=8,
                                                    num_heads=num_heads)
    out1 = model(batch)
    batch.bias = torch.randint(0, 2, (1, 5, 5), dtype=torch.bool)
    out2 = model(batch)
    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2)


def test_cls_token_transformation(simple_batch, transformer_model):
    """A learnable CLS token (`use_super_node`) must be *updated* by the first
    encoder layer.  We check that the value coming out of the full forward
    pass is no longer identical to the parameter that was fed in.
    """
    # ----- build graph & model ---------------------------------------------
    data = simple_batch(feat_dim=8, num_nodes=3)  # one graph → cls @ 0
    model = transformer_model(
        hidden_dim=8,
        encoder_cfg={
            **DEFAULT_ENCODER, "num_encoder_layers": 1,
            "use_super_node": True
        },
    )

    original_cls = model.cls_token.clone().squeeze(0)  # (C,)

    # ----- full forward pass -----------------------------------------------
    out = model(data)  # (N+1, C); cls is row 0
    cls_after = out[0]

    # ----- assertion --------------------------------------------------------
    assert not torch.allclose(cls_after, original_cls)


@pytest.mark.parametrize("seq_len,num_spatial,feat_dim", [(10, 4, 16)])
def test_spatial_bias_shifts_logits(spatial_batch, seq_len, num_spatial,
                                    feat_dim):
    out_channels = 2
    batch0 = spatial_batch(1, seq_len, num_spatial, feat_dim)
    batch1 = spatial_batch(1, seq_len, num_spatial, feat_dim)
    bias = torch.zeros((1, num_spatial, seq_len, seq_len))
    bias[:, :, 0, 1] = 5.0
    batch1.bias = bias
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 1}
    model = GraphTransformer(hidden_dim=feat_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    out0 = model(batch0)
    out1 = model(batch1)
    seq_len = batch0.x.size(0)
    assert out0.shape == (seq_len, out_channels)
    assert out1.shape == (seq_len, out_channels)
    assert not torch.allclose(out0, out1)


@pytest.mark.parametrize("use_super_node", [False])
def test_multi_provider_fusion(spatial_edge_batch, use_super_node):
    batch = spatial_edge_batch(2, seq_len=4, num_spatial=5, num_edges=6,
                               feat_dim=8)
    number_of_heads = 2
    provs = [
        GraphAttnSpatialBias(number_of_heads, 5, use_super_node),
        GraphAttnEdgeBias(number_of_heads, 6, use_super_node)
    ]
    encoder_cfg = {
        **DEFAULT_ENCODER, "use_super_node": use_super_node,
        "attn_bias_providers": provs,
        "num_heads": number_of_heads
    }
    model = GraphTransformer(
        hidden_dim=8,
        out_channels=2,
        encoder_cfg=encoder_cfg,
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

    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 1}
    gnn_cfg = {"gnn_block": gnn, "gnn_position": pos}
    model = GraphTransformer(hidden_dim=8, out_channels=2,
                             encoder_cfg=encoder_cfg, gnn_cfg=gnn_cfg)
    # patch the real encoder
    orig = model.encoder.layers[0].forward

    def track(x, mask, kp):
        seq.append("enc")
        return orig(x, mask, kp)

    model.encoder.layers[0].forward = track

    _ = model(simple_batch(8, 3))
    if pos == "pre":
        assert seq == ["gnn", "enc"]
    elif pos == "post":
        assert seq == ["enc", "gnn"]
    else:
        assert sorted(seq) == ["enc", "gnn"]


# ── Positional Encoder Hook Tests ───────────────────────────────────────────


def test_positional_encoders(simple_batch):
    batch = simple_batch(feat_dim=8, num_nodes=5)
    const_val1, const_val2 = 1.0, 2.0
    encoders = [
        ConstantDegreeEncoder(const_val1),
        ConstantDegreeEncoder(const_val2)
    ]
    encoder_cfg = {**DEFAULT_ENCODER, "positional_encoders": encoders}
    model = GraphTransformer(hidden_dim=8, out_channels=2,
                             encoder_cfg=encoder_cfg)
    # Store original features
    orig_x = batch.x.clone()
    # Run model
    model(batch)

    # Check if features were modified correctly before attention
    # By intercepting at encoder layer
    def check_hook(module, input, output):
        x = input[0]
        expected = orig_x + const_val1 + const_val2
        assert torch.allclose(x, expected)

    hook = model.encoder.register_forward_hook(check_hook)
    _ = model(batch)
    hook.remove()


def test_combined_positional_encoders(simple_batch):
    """Test GraphTransformer with degree, eigen and SVD positional encoders."""
    # Create a test batch with known structural features
    num_nodes = 4
    out_channels = 2
    hidden_dim = 16
    batch = simple_batch(feat_dim=16, num_nodes=num_nodes)
    batch.in_degree = torch.tensor([1, 2, 2, 1])
    batch.out_degree = torch.tensor([2, 1, 1, 2])
    batch.eig_pos_emb = torch.randn(num_nodes, 3)  # 3 eigenvectors
    batch.svd_pos_emb = torch.randn(num_nodes, 6)  # r=3 -> 6 features [U,V]
    # Create model with all three encoders
    encoders = [
        DegreeEncoder(num_in=3, num_out=3, hidden_dim=hidden_dim),
        EigEncoder(num_eigvec=3, hidden_dim=hidden_dim),
        SVDEncoder(r=3, hidden_dim=hidden_dim)
    ]
    encoder_cfg = {**DEFAULT_ENCODER, "positional_encoders": encoders}
    model = GraphTransformer(hidden_dim=hidden_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    # Verify basic shapes and execution
    out = model(batch)

    assert out.shape == (num_nodes, out_channels)
    # Verify encoders affect output
    # Remove one encoder and check outputs differ
    reduced_encoder_cfg = {
        **DEFAULT_ENCODER, "positional_encoders": encoders[:-1]
    }
    reduced = GraphTransformer(hidden_dim=hidden_dim,
                               out_channels=out_channels,
                               encoder_cfg=reduced_encoder_cfg)
    out2 = reduced(batch)
    assert not torch.allclose(out, out2)
    # Verify gradients flow through encoders
    loss = out.sum()
    loss.backward()
    for enc in encoders:
        for p in enc.parameters():
            assert p.grad is not None


# ── GNN‐Block Parametrized Tests ────────────────────────────────────────────


@pytest.mark.parametrize("conv,where", [(GCNConv, "pre"), (SAGEConv, "post"),
                                        (GATConv, "pre")])
def test_gnn_block_real_convs(simple_batch, conv, where):
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 2}
    base_out = GraphTransformer(hidden_dim=16, out_channels=4,
                                encoder_cfg=encoder_cfg)(simple_batch(16, 5))

    def gnn(data, x):
        return conv(x.size(-1), x.size(-1))(x, data.edge_index)

    gnn_cfg = {**DEFAULT_GNN, "gnn_block": gnn, "gnn_position": where}
    out = GraphTransformer(hidden_dim=16, out_channels=4,
                           encoder_cfg=encoder_cfg,
                           gnn_cfg=gnn_cfg)(simple_batch(16, 5))
    assert out.shape == base_out.shape
    assert not torch.allclose(out, base_out)


# ── Model Representation String ────────────────────────────────────────────


def dummy_gnn(data, x):
    return x


@pytest.mark.parametrize(
    "config, substrings",
    [
        (
            # default: no encoders
            {
                "hidden_dim": 16,
                "out_channels": 3,
                "encoder_cfg": {
                    **DEFAULT_ENCODER,
                    "use_super_node": False,
                    "num_encoder_layers": 0,
                    "attn_bias_providers": [],
                    "positional_encoders": [],
                },
                "gnn_cfg": {
                    "gnn_block": None,
                    "gnn_position": "pre",
                },
            },
            [
                "hidden_dim=16",
                "out_channels=3",
                "use_super_node=False",
                "num_encoder_layers=0",
                "bias_providers=[]",
                "pos_encoders=[]",
                "gnn_hook=None@'pre'",
            ],
        ),
        (
            # with all types of encoders
            {
                "hidden_dim": 32,
                "out_channels": 5,
                "encoder_cfg": {
                    **DEFAULT_ENCODER,
                    "use_super_node":
                    True,
                    "num_encoder_layers":
                    3,
                    "attn_bias_providers": [],
                    "positional_encoders": [
                        DegreeEncoder(3, 3, 32),
                        EigEncoder(4, 32),
                        SVDEncoder(3, 32),
                    ],
                },
                "gnn_cfg": {
                    **DEFAULT_GNN,
                    "gnn_block": None,
                    "gnn_position": "pre",
                },
            },
            [
                "hidden_dim=32",
                "out_channels=5",
                "use_super_node=True",
                "num_encoder_layers=3",
                "bias_providers=[]",
                "pos_encoders=['DegreeEncoder', 'EigEncoder', 'SVDEncoder']",
                "gnn_hook=None@'pre'",
            ],
        ),
        (
            # with single encoder and gnn hook
            {
                "hidden_dim": 8,
                "out_channels": 2,
                "encoder_cfg": {
                    **DEFAULT_ENCODER,
                    "use_super_node": False,
                    "num_encoder_layers": 1,
                    "attn_bias_providers": [],
                    "positional_encoders": [DegreeEncoder(2, 2, 8)],
                },
                "gnn_cfg": {
                    "gnn_block": dummy_gnn,
                    "gnn_position": "post",
                },
            },
            [
                "hidden_dim=8",
                "out_channels=2",
                "use_super_node=False",
                "num_encoder_layers=1",
                "bias_providers=[]",
                "pos_encoders=['DegreeEncoder']",
                "gnn_hook=dummy_gnn@'post'",
            ],
        ),
    ])
def test_repr_various_configs(config, substrings):
    model = GraphTransformer(
        hidden_dim=config["hidden_dim"],
        out_channels=config["out_channels"],
        encoder_cfg=config["encoder_cfg"],
        gnn_cfg=config["gnn_cfg"],
    )
    rep = repr(model)
    for sub in substrings:
        assert sub in rep, f"{sub!r} not found in repr: {rep}"


@pytest.mark.parametrize(
    "num_heads, dropout, ffn_hidden_dim, activation, "
    "providers, should_raise, err_keywords",
    [
        # valid
        (4, 0.0, None, "gelu", [], False, []),
        (2, 0.5, 16, "relu", [], False, []),
        # bad num_heads
        (0, 0.1, None, "gelu", [], True, ["num_heads"]),
        (-1, 0.1, None, "gelu", [], True, ["num_heads"]),
        # bad dropout
        (2, -0.1, None, "gelu", [], True, ["dropout", "[0,1)"]),
        (2, 1.0, None, "gelu", [], True, ["dropout", "[0,1)"]),
        # bad ffn_hidden_dim
        (2, 0.1, 4, "gelu", [], True, ["ffn_hidden_dim", "≥"]),
        # bad activation
        (2, 0.1, None, "invalid", [], True, ["not", "supported"]),
        # mismatched bias‐provider heads
        (2, 0.1, None, "gelu", [GraphAttnSpatialBias(3, 5)
                                ], True, ["BiasProvider"]),
    ],
    ids=[
        "valid-defaults",
        "valid-relu",
        "bad-heads-zero",
        "bad-heads-negative",
        "bad-dropout-neg",
        "bad-dropout-eq1",
        "bad-ffn-too-small",
        "bad-activation",
        "bad-bias-heads",
    ])
def test_transformer_init_arg_validation(num_heads, dropout, ffn_hidden_dim,
                                         activation, providers, should_raise,
                                         err_keywords):
    encoder_cfg = {
        "num_encoder_layers": 1,
        "num_heads": num_heads,
        "dropout": dropout,
        "ffn_hidden_dim": ffn_hidden_dim,
        "activation": activation,
    }
    if providers:
        encoder_cfg["attn_bias_providers"] = providers
    if should_raise:
        with pytest.raises(ValueError) as exc:
            GraphTransformer(hidden_dim=8, out_channels=2,
                             encoder_cfg=encoder_cfg)
        msg = str(exc.value).lower()
        for kw in err_keywords:
            assert kw.lower() in msg
    else:
        model = GraphTransformer(hidden_dim=8, out_channels=2,
                                 encoder_cfg=encoder_cfg)
        # API‐level sanity
        assert model.num_heads == num_heads
        assert pytest.approx(model.dropout, rel=1e-6) == dropout
        expected_ffn = (ffn_hidden_dim if ffn_hidden_dim is not None else 4 *
                        8)
        assert model.ffn_hidden_dim == expected_ffn
        assert model.activation == activation
        # layer‐level sanity
        layer = (model.encoder
                 if not model.is_encoder_stack else model.encoder[0])
        assert isinstance(layer, GraphTransformerEncoderLayer)
        assert layer.num_heads == num_heads
        assert pytest.approx(layer.dropout_layer.p, rel=1e-6) == dropout
        assert layer.ffn_hidden_dim == expected_ffn


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if CUDA else []))
def test_parameters_and_buffers_on_same_device(
    transformer_model,
    full_feature_batch,
    structural_config,
    positional_config,
    degree_encoder,
    eig_encoder,
    svd_encoder,
    device,
):
    # skip gpu if it's not actually there
    if device == 'cuda' and not CUDA:
        pytest.skip("CUDA not available")
    # build positional encoders
    pos_encoders = [
        degree_encoder(
            positional_config['max_degree'],
            positional_config['max_degree'],
            structural_config['feat_dim'],
        ),
        eig_encoder(
            positional_config['num_eigvec'],
            structural_config['feat_dim'],
        ),
        svd_encoder(
            positional_config['num_sing_vec'],
            structural_config['feat_dim'],
        ),
    ]
    # build bias providers
    bias_providers = [
        GraphAttnSpatialBias(
            num_heads=2,
            num_spatial=positional_config['num_spatial'],
            use_super_node=True,
        ),
        GraphAttnEdgeBias(
            num_heads=2,
            num_edges=positional_config['num_edges'],
            use_super_node=True,
        ),
    ]
    encoder_cfg = {
        "use_super_node": True,
        "num_encoder_layers": 2,
        "num_heads": 2,
        "positional_encoders": pos_encoders,
        "attn_bias_providers": bias_providers,
    }
    # instantiate and move to target device
    model = transformer_model(
        hidden_dim=structural_config['feat_dim'],
        out_channels=3,
        encoder_cfg=encoder_cfg,
    ).to(device)
    # build a batch that lives on the same device
    batch = full_feature_batch(device)
    # do one forward pass (in case any buffers are lazily created)
    _ = model(batch)
    # all parameters must be on `device`
    for name, param in model.named_parameters():
        assert param.device.type == device, (
            f"Parameter '{name}' is on {param.device}, expected {device}")
    # all buffers (e.g. cls_token, layernorm stats, embeddings, etc.) too
    for name, buf in model.named_buffers():
        assert buf.device.type == device, (
            f"Buffer '{name}' is on {buf.device}, expected {device}")


# ── Edge Case Tests ──────────────────────────────────────────────────────────


def test_graph_transformer_none_features_handled(simple_none_batch):
    """Test GraphTransformer regression: handling None node features.

    test that GraphTransformer can process datasets with
    None node features (like QM7B regression dataset).

    Parameters
    ----------
    simple_none_batch : Batch
        A batch containing data with x=None to simulate missing node features.
    """
    hidden_dim = 16
    out_channels = 2
    input_feat_dim = 10

    encoder_cfg = {
        "node_feature_encoder":
        nn.Sequential(nn.Linear(input_feat_dim, hidden_dim), nn.ReLU())
    }
    model = GraphTransformer(hidden_dim=hidden_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    output = model(simple_none_batch)
    num_nodes = simple_none_batch.num_nodes
    assert output.shape == (num_nodes, out_channels)
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


@pytest.mark.parametrize(
    "encoder_factory,feat_dim,should_raise",
    [
        # Linear with mismatched input features
        (lambda: nn.Linear(10, 64), 1, True),
        # Sequential with mismatched input features
        (lambda: nn.Sequential(nn.Linear(10, 64), nn.ReLU()), 1, True),
        # Linear with matching input features
        (lambda: nn.Linear(1, 64), 1, False),
        # Sequential with matching input features
        (lambda: nn.Sequential(nn.Linear(1, 64), nn.ReLU()), 1, False),
        # Identity with mismatched input features (should raise)
        (lambda: nn.Identity(), 1, True),
        # Identity with matching input features (should not raise)
        (lambda: nn.Identity(), 64, False),
    ],
    ids=[
        "linear_mismatch",
        "sequential_mismatch",
        "linear_match",
        "sequential_match",
        "identity_mismatch",
        "identity_match",
    ])
def test_graph_transformer_feature_dim_mismatch(simple_batch, encoder_factory,
                                                feat_dim, should_raise):
    batch = simple_batch(feat_dim=feat_dim, num_nodes=10)
    encoder_cfg = {"node_feature_encoder": encoder_factory()}
    model = GraphTransformer(hidden_dim=64, out_channels=1,
                             encoder_cfg=encoder_cfg)
    if should_raise:
        with pytest.raises(ValueError,
                           match="Node feature dimension mismatch"):
            model(batch)
    else:
        model(batch)


def _clone_learnable(model):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def test_reset_parameters_changes_weights():
    torch.manual_seed(42)
    model = GraphTransformer(hidden_dim=8, out_channels=2)
    old_params = _clone_learnable(model)
    torch.manual_seed(43)
    model.reset_parameters()
    new_params = _clone_learnable(model)
    # At least one parameter should have changed
    assert any((o != n).any() for o, n in zip(old_params, new_params))


@pytest.mark.parametrize(
    ("cache_masks", "expected_calls"),
    [
        (True, 1),  # caching ON  → `build_key_padding` called once
        (False, 2),  # caching OFF → called on every forward
    ],
)
def test_attention_mask_caching(simple_batch, cache_masks, expected_calls,
                                patched_build_key_padding):
    """Verify that key-padding masks are cached only when requested."""
    batch = simple_batch(feat_dim=8, num_nodes=5)

    model = GraphTransformer(
        hidden_dim=8,
        out_channels=2,
        encoder_cfg={"num_encoder_layers": 1},
        cache_masks=cache_masks,
    )

    model(batch)
    model(batch)

    assert patched_build_key_padding.call_count == expected_calls


@pytest.mark.parametrize(
    "case_id, model_kwargs, run_fn, expected_calls",
    [
        # 1) same batch twice  → only first call builds the mask
        (
            "same_shape_cached",
            {
                "cache_masks": True,
                "encoder_cfg": {
                    "num_encoder_layers": 1
                }
            },
            lambda m, make: [m(make(8, 5)), m(make(8, 5))],
            1,
        ),
        # 2) different #nodes between passes → needs a new mask
        (
            "diff_shape_cached",
            {
                "cache_masks": True,
                "encoder_cfg": {
                    "num_encoder_layers": 1
                }
            },
            lambda m, make: [m(make(8, 5)), m(make(8, 7))],
            2,
        ),
        # 3) two encoder layers with different head counts (4 & 8) in one pass
        (
            "diff_heads_layer",
            {
                "cache_masks": True,
                "encoder_cfg": {
                    "num_encoder_layers": 2
                }
            },
            lambda m, make: [
                # re-wire second layer to have 8 heads
                setattr(
                    m.encoder[1],
                    "__dict__",
                    GraphTransformerEncoderLayer(hidden_dim=8, num_heads=8,
                                                 dropout=0.0).__dict__,
                ),
                m(make(8, 5)),
            ],
            2,
        ),
        # 4) cache cleared via reset_parameters()
        (
            "reset_clears_cache",
            {
                "cache_masks": True,
                "encoder_cfg": {
                    "num_encoder_layers": 1
                }
            },
            lambda m, make:
            [m(make(8, 5)), m.reset_parameters(),
             m(make(8, 5))],
            2,
        ),
        # 5) super-node path should still respect caching
        (
            "super_node",
            {
                "cache_masks": True,
                "encoder_cfg": {
                    "num_encoder_layers": 1,
                    "use_super_node": True
                }
            },
            lambda m, make: [m(make(8, 5)), m(make(8, 5))],
            1,
        ),
    ],
)
def test_attention_mask_cache_edge_cases(simple_batch, case_id, model_kwargs,
                                         run_fn, expected_calls,
                                         patched_build_key_padding):
    """Validates attention-mask caching behaviour
    across various scenarios, ensuring correct call counts
    to `build_key_padding` function.
    """
    batch_maker = simple_batch
    model = GraphTransformer(hidden_dim=8, out_channels=2, **model_kwargs)
    _ = run_fn(model, batch_maker)

    assert (patched_build_key_padding.call_count == expected_calls
            ), f"{case_id}: expected {expected_calls}, got \
        {patched_build_key_padding.call_count}"


def test_no_head_output_hidden_dim(simple_batch):
    """Test that the output hidden dimension matches the model's hidden_dim."""
    feat_dim = 8
    hidden_dim = 16
    out_channels = None
    num_nodes = 5
    batch = simple_batch(feat_dim=feat_dim, num_nodes=num_nodes)
    encoder_cfg = {**DEFAULT_ENCODER, "num_encoder_layers": 1}
    model = GraphTransformer(hidden_dim=hidden_dim, out_channels=out_channels,
                             encoder_cfg=encoder_cfg)
    out = model(batch)
    assert out.shape[0] == num_nodes, (
        f"Expected output batch size {num_nodes}, got {out.shape[0]}")
    assert out.shape[1] == model.hidden_dim, (
        f"Expected output hidden dim {model.hidden_dim}, got {out.shape[1]}")


def test_collect_attn_bias_dtype_behavior(simple_batch):
    """Test _collect_attn_bias returns correct dtype based on cast_bias."""
    # Prepare a batch with a simple bias mask
    batch = simple_batch(feat_dim=8, num_nodes=4)
    batch.bias = torch.ones((1, 1, 4, 4), dtype=torch.float32)

    # Default: cast_bias=False → always float32
    model = GraphTransformer(hidden_dim=8, out_channels=2, cast_bias=False)
    bias_out = model._collect_attn_bias(batch)
    assert bias_out.dtype == torch.float32

    # Explicit cast_bias=True on float32 x still yields float32
    model.cast_bias = True
    bias_out2 = model._collect_attn_bias(batch)
    assert bias_out2.dtype == torch.float32

    # Simulate float16 features
    batch.x = batch.x.to(torch.float16)
    model.cast_bias = False
    bias_out3 = model._collect_attn_bias(batch)
    assert bias_out3.dtype == torch.float32

    # With cast_bias=True on float16 x, we get float16
    model.cast_bias = True
    bias_out4 = model._collect_attn_bias(batch)
    assert bias_out4.dtype == torch.float16


@pytest.mark.skipif(not CUDA,
                    reason="CUDA required for mixed-precision AMP test")
def test_mixed_precision_amp_stability(simple_batch):
    """Ensure forward/backward under AMP float16 runs without NaNs."""
    batch = simple_batch(feat_dim=8, num_nodes=5).to('cuda')
    batch.bias = torch.zeros((1, 1, 5, 5), device='cuda')

    model = GraphTransformer(hidden_dim=8, out_channels=2, cast_bias=False)
    model = model.to('cuda').train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9,
                                weight_decay=1e-4)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(batch)
        loss = out.sum()

    optimizer.zero_grad()
    loss.backward()

    # No NaNs/Infs in output
    assert torch.isfinite(out).all()

    # No NaNs/Infs in any gradient
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

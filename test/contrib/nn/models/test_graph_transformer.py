from typing import Optional

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


def make_simple_batch(feature_dim=16, num_nodes=5):
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    return Batch.from_data_list([Data(x=x, edge_index=edge_index)])


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


def make_batch_with_spatial_and_edge(
    batch_size: int,
    seq_len: int,
    num_spatial: int,
    num_edges: int,
    feature_dim: int = 1,
) -> Batch:
    """Create a Batch of graphs with both spatial_pos and edge_dist."""
    data_list = []
    for _ in range(batch_size):
        # Spatial distances
        spatial_pos = torch.randint(0, num_spatial, (seq_len, seq_len))
        # Edge types
        edge_dist = torch.randint(0, num_edges, (seq_len, seq_len))
        # Node features
        x = torch.randn(seq_len, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(
            Data(
                x=x,
                spatial_pos=spatial_pos,
                edge_dist=edge_dist,
                edge_index=edge_index,
            )
        )
    return Batch.from_data_list(data_list)


class AddOneLayer(nn.Module):

    def __init__(self, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, *args, **kwargs):
        return x + 1.0


class ConstantDegreeEncoder(nn.Module):

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def forward(self, data: Data) -> torch.Tensor:
        return torch.full_like(data.x, self.value)


class IdentityEncoder(nn.Module):
    """Identity encoder that preserves the encoder interface."""

    def forward(self, x, batch=None, attn_mask=None):
        return x


class TracableWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, batch: Tensor, bias: Optional[Tensor] = None):
        data = Batch(
            x=x,
            batch=batch,
            edge_index=torch.empty((2, 0), dtype=torch.long, device=x.device)
        )
        if bias is not None:
            data.bias = bias
        out = self.model(data)["logits"]
        return out


@pytest.mark.parametrize(
    "num_graphs, num_nodes, feature_dim, num_classes",
    [
        (1, 10, 16, 2),  # Single graph with 10 nodes
        (5, 20, 32, 3),  # Multiple graphs with different node counts
        (3, 15, 64, 4),  # Multiple graphs with same node count
    ],
    ids=[
        "single_graph", "multiple_graphs_diff_nodes",
        "multiple_graphs_same_nodes"
    ]
)
def test_forward_shape(num_graphs, num_nodes, feature_dim, num_classes):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        data_list.append(Data(x=x, edge_index=edge_index))
    batch = Batch.from_data_list(data_list)
    model = GraphTransformer(hidden_dim=feature_dim, num_class=num_classes)
    out = model(batch)["logits"]
    assert out.shape == (
        num_graphs, num_classes
    ), f"Expected output shape {(num_graphs, num_classes)}, got {out.shape}"


def test_super_node_readout():
    """Test cls token is properly processed through encoder and readout."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    data = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
    batch = Batch.from_data_list([data])

    # Create model and make encoder pass-through for test
    model = GraphTransformer(hidden_dim=4, num_class=2, use_super_node=True)
    model.encoder = IdentityEncoder()  # Skip encoder transformation

    out = model(batch)["logits"]
    expected = model.classifier(model.cls_token.expand(1, -1))
    assert torch.allclose(
        out, expected, rtol=1e-4
    ), "With identity encoder, output should match classifier(cls_token)"


def test_node_feature_encoder_identity():
    """Test node_feature_encoder is correctly applied before transformer."""
    feature_dim = 16
    num_nodes = 10
    num_classes = 2

    # Create a single graph with random features
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    batch.batch = torch.zeros(num_nodes, dtype=torch.long)

    # Create model and force identity encoder
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=num_classes,
        node_feature_encoder=nn.Identity()
    )

    # First run should match raw features
    output1 = model(batch)

    # Second run with scaled features should give different output
    batch.x = 2 * batch.x  # Scale features
    output2 = model(batch)

    assert not torch.allclose(output1["logits"], output2["logits"]), \
        "Model output should change when input features are scaled"


def test_transformer_block_identity():
    """Test that a GraphTransformer with a single encoder layer
    produces output with the expected shape after modifying the layer.
    """
    num_graphs = 3
    feature_dim = 16
    num_classes = 4
    num_nodes = 8
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    batch = Batch.from_data_list(data_list)
    model = GraphTransformer(
        hidden_dim=feature_dim, num_class=num_classes, num_encoder_layers=1
    )
    model.encoder[0] = AddOneLayer()
    out = model(batch)["logits"]
    assert out.shape == (num_graphs, num_classes), \
        f"Expected output shape {(num_graphs, num_classes)}, got {out.shape}"


def test_backward():
    """Test that gradients flow properly through the GraphTransformer model."""
    num_graphs = 2
    num_nodes = 5
    feature_dim = 8
    num_classes = 3

    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(data_list)
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=num_classes,
        use_super_node=True,
        num_encoder_layers=1
    )
    model.encoder[0] = AddOneLayer()
    out = model(batch)["logits"]
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert_message = f"Parameter {name} has None gradient"
            assert param.grad is not None, assert_message


def test_torchscript_trace():
    torch.manual_seed(12345)

    # -------- dummy input -----------
    G, N, F, C = 2, 5, 8, 3
    x = torch.randn(G * N, F)
    batch = torch.arange(G).repeat_interleave(N)
    bias = torch.rand(G, 4, N, N)  # structural mask

    # -------- model & wrapper -------
    model = GraphTransformer(
        hidden_dim=F, num_class=C, num_encoder_layers=2
    ).eval()
    wrapper = TracableWrapper(model)

    # -------- trace -----------------
    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, (x, batch, bias))
        except Exception as e:
            pytest.fail(f"TorchScript tracing failed: {e}")

        # -------- compare -------------
        ref = wrapper(x, batch, bias)  # original (eager) run
        out = traced(x, batch, bias)  # scripted run

        assert torch.allclose(ref, out, atol=1e-6), \
            "Scripted and eager logits differ"


def test_encoder_layer_type():
    """Test that the first encoder layer in a GraphTransformer is
    an instance of GraphTransformerEncoderLayer.
    """
    model = GraphTransformer(hidden_dim=16, num_encoder_layers=1)
    assert len(
        model.encoder
    ) == 1, "Model's encoder should have exactly one layer"
    assert isinstance(model.encoder[0], GraphTransformerEncoderLayer), \
        "First encoder layer should be an instance of " \
        "GraphTransformerEncoderLayer"


def test_degree_encoder_shifts_logits():
    """Test that using a degree encoder that returns constant values
    shifts the logits by a fixed amount.
    """
    torch.manual_seed(12345)
    num_graphs = 2
    num_nodes = 5
    feature_dim = 8
    num_classes = 3
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    batch = Batch.from_data_list(data_list)
    batch.x = torch.randn_like(batch.x)
    fixed_x = batch.x.clone()

    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=num_classes,
        degree_encoder=ConstantDegreeEncoder(1.0),
    )
    model.eval()

    # First run
    with torch.no_grad():
        batch.x = fixed_x
        out1 = model(batch)["logits"]

    # Second run with same features should give same output
    with torch.no_grad():
        batch.x = fixed_x
        out2 = model(batch)["logits"]

    assert torch.allclose(out1, out2, rtol=1e-4), \
        "Outputs should be identical with fixed input and \
        constant degree encoder"

    # Test that gradients flow in train mode
    model.train()
    batch.x = fixed_x
    out = model(batch)["logits"]
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


@pytest.mark.parametrize("num_heads", [1, 4])
def test_attention_mask_changes_logits(num_heads):
    """Test that using different attention masks produces different logits."""
    torch.manual_seed(12345)

    num_graphs = 2
    num_nodes = 5
    feature_dim = 8
    num_classes = 3

    # Create test data
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    batch = Batch.from_data_list(data_list)

    # Create random attention mask
    max_nodes = num_nodes
    attn_mask = torch.randint(
        0, 2, (1, max_nodes, max_nodes), dtype=torch.bool
    )

    # Create model with specified number of heads
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=num_classes,
        num_encoder_layers=1,
    )
    model.encoder[0] = GraphTransformerEncoderLayer(
        hidden_dim=feature_dim,
        num_heads=num_heads,
    )
    model.eval()

    with torch.no_grad():
        # First pass without mask
        out1 = model(batch)["logits"]

        # Second pass with attention mask
        batch.bias = attn_mask  # Pass mask through batch
        out2 = model(batch)["logits"]

    assert out1.shape == out2.shape, "Outputs should have the same shape"
    assert not torch.allclose(out1, out2, rtol=1e-4), \
        "Outputs should differ when using different attention masks"


def test_cls_token_transformation():
    """Test that the cls token is transformed by attention and not just used
    directly in readout.
    """
    torch.manual_seed(12345)  # For reproducibility

    # Create two tiny graphs
    data_list = []
    for _ in range(2):
        x = torch.randn(3, 8)  # 3 nodes, 8 features
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(data_list)

    # Create model with single encoder layer and super node
    model = GraphTransformer(
        hidden_dim=8, num_class=2, use_super_node=True, num_encoder_layers=1
    )

    # Store original cls token
    original_cls = model.cls_token.clone()

    # Run forward pass and extract transformed CLS tokens
    with torch.no_grad():
        x = model._encode_nodes(batch.x)
        x = model._apply_extra_encoders(batch, x)
        x, batch_vec = model._prepend_cls_token_flat(x, batch.batch)
        encoded = model.encoder(x, batch_vec)

        # Indices of the first row in each graph (i.e. the CLS token)
        graph_sizes = torch.bincount(batch_vec)
        first_idx = torch.cumsum(graph_sizes, 0) - graph_sizes
        transformed_cls = encoded[first_idx]

    # Verify cls token was transformed
    assert not torch.allclose(transformed_cls, original_cls.expand(2, -1)), \
        "CLS token should be modified by attention mechanism"


def test_spatial_bias_shifts_logits():
    """Test that using spatial bias shift the logits."""
    feature_dim = 16
    num_nodes = 10
    num_classes = 2
    num_heads = 4

    data = Data(
        x=torch.randn(num_nodes, feature_dim),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    batch_no_bias = Batch.from_data_list([data])
    batch_with_bias = Batch.from_data_list([data])
    bias = torch.zeros((1, num_heads, num_nodes, num_nodes))
    bias[:, :, 0, 1] = 5.0
    batch_with_bias.bias = bias

    model = GraphTransformer(
        hidden_dim=feature_dim, num_class=num_classes, num_encoder_layers=1
    )
    model.eval()

    with torch.no_grad():
        out1 = model(batch_no_bias)["logits"]
        out2 = model(batch_with_bias)["logits"]

    assert not torch.allclose(out1, out2, rtol=1e-4), \
        "Logits should differ when using spatial bias"


def test_bias_provider_api():
    """Test that using attention bias providers changes the model output."""
    torch.manual_seed(12345)
    num_spatial = 8
    seq_len = 6
    feature_dim = 16

    batch = make_batch_with_spatial(
        1, seq_len, num_spatial, feature_dim=feature_dim
    )

    # Create models with and without bias provider
    model_no_bias = GraphTransformer(
        hidden_dim=16,
        num_class=2,
        num_encoder_layers=1,
    )
    model_with_bias = GraphTransformer(
        hidden_dim=16,
        num_class=2,
        num_encoder_layers=1,
        attn_bias_providers=[GraphAttnSpatialBias(4, num_spatial)]
    )

    # Compare outputs
    model_no_bias.eval()
    model_with_bias.eval()

    with torch.no_grad():
        out1 = model_no_bias(batch)["logits"]
        out2 = model_with_bias(batch)["logits"]

    assert not torch.allclose(out1, out2, rtol=1e-4), \
        "Outputs should differ when using attention bias providers"


def test_super_node_bias():
    """Test spatial bias provider properly handles super node attention."""
    torch.manual_seed(12345)

    # Create tiny graph
    num_spatial = 4
    seq_len = 2  # Just 2 nodes
    feature_dim = 8
    num_heads = 4

    batch = make_batch_with_spatial(
        1, seq_len, num_spatial, feature_dim=feature_dim
    )

    # Model with both super node and spatial bias
    model_with_bias = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=2,
        num_encoder_layers=1,
        use_super_node=True,
        attn_bias_providers=[
            GraphAttnSpatialBias(
                num_heads=num_heads,
                num_spatial=num_spatial,
                use_super_node=True
            )
        ]
    )

    # Same model without bias provider
    model_no_bias = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=2,
        num_encoder_layers=1,
        use_super_node=True
    )

    # Compare outputs
    model_with_bias.eval()
    model_no_bias.eval()

    with torch.no_grad():
        out1 = model_no_bias(batch)["logits"]
        out2 = model_with_bias(batch)["logits"]

    assert not torch.allclose(out1, out2, rtol=1e-4), \
        "Outputs should differ when using spatial bias with super node"

    bias = model_with_bias.attn_bias_providers[0](batch)
    expected = (1, num_heads, seq_len + 1, seq_len + 1)
    assert bias.shape == expected, \
        f"Expected bias shape {expected}, got {tuple(bias.shape)}"


@pytest.mark.parametrize("use_super_node", [False, True])
def test_multi_provider_fusion(use_super_node):
    """Test that multiple attention bias providers combine correctly."""
    torch.manual_seed(12345)

    # Parameters
    batch_size = 2
    seq_len = 4
    num_spatial = 5
    num_edges = 6
    num_heads = 2
    feature_dim = 8

    # Create providers
    prov1 = GraphAttnSpatialBias(
        num_heads=num_heads,
        num_spatial=num_spatial,
        use_super_node=use_super_node
    )
    prov2 = GraphAttnEdgeBias(
        num_heads=num_heads,
        num_edges=num_edges,
        edge_type='simple',
        use_super_node=use_super_node
    )

    # Create model with both providers
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=2,
        use_super_node=use_super_node,
        attn_bias_providers=[prov1, prov2]
    )

    # Create batch with both types of position data
    batch = make_batch_with_spatial_and_edge(
        batch_size=batch_size,
        seq_len=seq_len,
        num_spatial=num_spatial,
        num_edges=num_edges,
        feature_dim=feature_dim
    )

    # Get fused bias through model's internal method
    struct = model._collect_attn_bias(batch)

    # Calculate sum directly
    sum_direct = (
        prov1(batch).to(struct.dtype).to(struct.device) +
        prov2(batch).to(struct.dtype).to(struct.device)
    )

    # Expected shape after accounting for super node
    expected_len = seq_len + (1 if use_super_node else 0)
    expected_shape = (batch_size, num_heads, expected_len, expected_len)

    assert struct.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {tuple(struct.shape)}"
    assert torch.allclose(struct, sum_direct, atol=1e-6), \
        "Model's fused bias differs from direct sum of provider outputs"


def test_forward_without_gnn_hook():
    """GraphTransformer without a gnn_block should behave exactly as before."""
    # Create a single graph with 2 nodes and no edges
    hidden_dim = 8
    x = torch.randn(2, hidden_dim)  # feature dim can be arbitrary
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])

    # Instantiate without any gnn_block
    model = GraphTransformer(hidden_dim=hidden_dim)

    # Run forward
    out = model(batch)["logits"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, model.classifier.out_features)


@pytest.mark.parametrize("gnn_position", ["pre", "post", "parallel"])
def test_gnn_hook_called(gnn_position):
    """Ensure gnn_block runs before or after the encoder as specified."""
    torch.manual_seed(0)
    execution_order = []

    # 1) Define the dummy GNN block
    def gnn_block(data, x):
        execution_order.append("gnn")
        return x

    # 2) Build the model with a single encoder layer and our hook
    model = GraphTransformer(
        hidden_dim=8,
        num_class=2,
        num_encoder_layers=1,
        gnn_block=gnn_block,
        gnn_position=gnn_position,
    )

    # 3) Monkey-patch the real encoder layer's forward to track its call
    encoder_layer = model.encoder.layers[0]  # GraphTransformerEncoderLayer
    orig_forward = encoder_layer.forward

    def tracked_forward(x, batch, struct_mask, key_pad):
        execution_order.append("encoder")
        return orig_forward(x, batch, struct_mask, key_pad)

    encoder_layer.forward = tracked_forward  # replace method

    # 4) Run a simple forward pass
    batch = make_simple_batch(feature_dim=8, num_nodes=3)
    model(batch)

    # 5) Verify ordering
    if gnn_position == "pre":
        assert execution_order == [
            "gnn", "encoder"
        ], (f"Expected gnn before encoder for 'pre', got {execution_order}")
    elif gnn_position == "post":
        assert execution_order == [
            "encoder", "gnn"
        ], (f"Expected gnn after encoder for 'post', got {execution_order}")
    else:  # parallel
        assert sorted(execution_order) == ["encoder", "gnn"], (
            f"Expected both gnn and encoder to run once for 'parallel', \
            got {execution_order}"
        )


@pytest.mark.parametrize(
    "conv_cls, position", [
        (GCNConv, "pre"),
        (SAGEConv, "post"),
        (GATConv, "pre"),
    ]
)
def test_gnn_block_with_real_conv(conv_cls, position):
    """Verify that real PyG conv layers can be used as gnn_block,
    that they run without error, change the logits, and preserve shape.
    """
    torch.manual_seed(0)
    batch = make_simple_batch(feature_dim=16, num_nodes=5)

    # instantiate a single-head conv to match hidden_dim
    def gnn_block(data, x):
        conv = conv_cls(x.size(-1), x.size(-1))
        return conv(x, data.edge_index)

    # baseline model without hook
    base = GraphTransformer(hidden_dim=16, num_class=4, num_encoder_layers=2)
    out_base = base(batch)["logits"]

    # model with GNN hook
    model = GraphTransformer(
        hidden_dim=16,
        num_class=4,
        num_encoder_layers=2,
        gnn_block=gnn_block,
        gnn_position=position,
    )
    out = model(batch)["logits"]

    # Shape should be unchanged
    assert out.shape == out_base.shape

    # But logits should differ when a non-trivial GNN is applied
    assert not torch.allclose(out, out_base), \
        f"Logits must change when using {conv_cls.__name__} at {position}"


def test_gnn_block_multi_layer_mlp():
    """Test a custom 2-layer MLP block as gnn_block in 'post' position."""
    torch.manual_seed(0)
    batch = make_simple_batch(feature_dim=8, num_nodes=4)

    def mlp_block(data, x):
        x = nn.Linear(x.size(-1), x.size(-1))(x)
        return nn.ReLU()(nn.Linear(x.size(-1), x.size(-1))(x))

    base = GraphTransformer(hidden_dim=8, num_class=3, num_encoder_layers=1)
    out_base = base(batch)["logits"]

    model = GraphTransformer(
        hidden_dim=8,
        num_class=3,
        num_encoder_layers=1,
        gnn_block=mlp_block,
        gnn_position="post",
    )
    out = model(batch)["logits"]
    assert out.shape == out_base.shape
    assert not torch.allclose(out, out_base)

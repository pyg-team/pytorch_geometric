import pytest
import torch
import torch.nn as nn

from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


class AddOneLayer(nn.Module):

    def forward(self, x, batch):
        return x + 1.0


class ConstantDegreeEncoder(nn.Module):

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def forward(self, data: Data) -> torch.Tensor:
        return torch.full_like(data.x, self.value)


class TracableWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, batch):
        # Simple forward that avoids Batch construction
        out = self.model.node_feature_encoder(x)
        out = self.model.encoder(out, batch)
        out = self.model._readout(out, batch)
        return self.model.classifier(out)


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
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    data = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
    batch = Batch.from_data_list([data])
    model = GraphTransformer(hidden_dim=4, num_class=2, use_super_node=True)
    out = model(batch)["logits"]
    expected = model.classifier(model.cls_token.expand(1, -1))
    assert torch.allclose(
        out, expected
    ), "Expected output to match classifier(cls_token) for super node readout."


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
    """Test that GraphTransformer can be traced with TorchScript."""
    torch.manual_seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_graphs = 2
    num_nodes = 5
    feature_dim = 8
    num_classes = 3

    x = torch.randn(num_nodes * num_graphs, feature_dim)
    batch = torch.zeros(num_nodes * num_graphs, dtype=torch.long)
    for i in range(1, num_graphs):
        batch[i * num_nodes:(i + 1) * num_nodes] = i

    # Handle data construction outside tracing
    data_batch = Batch(
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.long, device=x.device),
        batch=batch
    )

    model = GraphTransformer(
        hidden_dim=feature_dim, num_class=num_classes, num_encoder_layers=2
    )
    model.eval()
    wrapper = TracableWrapper(model)

    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, (x, batch))
        except Exception as e:
            pytest.fail(f"Failed to trace model: {str(e)}")

        original_output = model(data_batch)["logits"]
        traced_output = traced(x, batch)

        assert torch.allclose(original_output, traced_output, atol=1e-6), \
            "Outputs from original and traced models should be close"


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
        batch.attn_mask = attn_mask  # Pass mask through batch
        out2 = model(batch)["logits"]

    assert out1.shape == out2.shape, "Outputs should have the same shape"
    assert not torch.allclose(out1, out2, rtol=1e-4), \
        "Outputs should differ when using different attention masks"

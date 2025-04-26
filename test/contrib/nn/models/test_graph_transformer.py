# isort: skip_file
# yapf: disable
import pytest
import torch
import torch.nn as nn

from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


class AddOneLayer(nn.Module):
    def forward(self, x, batch):
        return x + 1.0


def scale_encoder(x):
    return x * 2


class TracableWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, batch):
        return self.model.forward(Batch(x=x, batch=batch))["logits"]


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
    ])
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
    """Test that node_feature_encoder correctly scales
    inputs and affects logits.
    """
    feature_dim = 16
    num_nodes = 10
    num_classes = 2

    # Create a single graph with random features
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    batch.batch = torch.zeros(num_nodes, dtype=torch.long)
    model = GraphTransformer(hidden_dim=feature_dim, num_class=num_classes,
                             node_feature_encoder=scale_encoder)

    output = model(batch)["logits"]
    scaled_x = scale_encoder(x)
    pooled_x = global_mean_pool(scaled_x, batch.batch)
    expected_output = model.classifier(pooled_x)
    assert torch.allclose(output, expected_output), \
        "Model output with encoder does not match expected computation"


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
    model = GraphTransformer(hidden_dim=feature_dim, num_class=num_classes,
                             num_encoder_layers=1)
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
    model = GraphTransformer(hidden_dim=feature_dim, num_class=num_classes,
                             use_super_node=True, num_encoder_layers=1)
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
    num_graphs = 2
    num_nodes = 5
    feature_dim = 8
    num_classes = 3
    x = torch.randn(num_nodes * num_graphs, feature_dim)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(num_nodes * num_graphs, dtype=torch.long)
    for i in range(1, num_graphs):
        batch[i * num_nodes:(i + 1) * num_nodes] = i
    data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
    model = GraphTransformer(
        hidden_dim=feature_dim,
        num_class=num_classes,
    )
    wrapper = TracableWrapper(model)
    traced = torch.jit.trace(wrapper, (x, batch))
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
        model.encoder) == 1, "Model's encoder should have exactly one layer"
    assert isinstance(model.encoder[0], GraphTransformerEncoderLayer), \
        "First encoder layer should be an instance of " \
        "GraphTransformerEncoderLayer"
# yapf: enable

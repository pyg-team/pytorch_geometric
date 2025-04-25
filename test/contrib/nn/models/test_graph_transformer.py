import pytest
import torch

from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


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

    def scale_encoder(x):
        return x * 2

    # Create model with the encoder
    model = GraphTransformer(hidden_dim=feature_dim, num_class=num_classes,
                             node_feature_encoder=scale_encoder)

    output = model(batch)["logits"]
    scaled_x = scale_encoder(x)
    pooled_x = global_mean_pool(scaled_x, batch.batch)
    expected_output = model.classifier(pooled_x)
    assert torch.allclose(output, expected_output), \
        "Model output with encoder does not match expected computation"

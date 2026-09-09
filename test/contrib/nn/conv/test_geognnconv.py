import pytest
import torch

from torch_geometric.contrib.nn.conv.geognn_conv import (
    GeoGNNConv,
    GraphNorm,
    ModGINConv,
)


@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_graph_norm(batch_size):
    norm = GraphNorm()

    num_nodes_per_graph = 10
    total_nodes = batch_size * num_nodes_per_graph

    # Simulate input features (x) - random values for the sake of testing
    x = torch.randn(total_nodes, 5)

    # Create batch vector
    batch = torch.cat(
        [torch.full((num_nodes_per_graph, ), i) for i in range(batch_size)])

    # Apply GraphNorm
    normalized_x = norm(x, batch)

    # Check if the normalization is done correctly
    for i in range(batch_size):
        batch_indices = (batch == i)
        sqrt_num_nodes = num_nodes_per_graph**0.5
        expected_output = x[batch_indices] / sqrt_num_nodes

        # Compare the actual output with the expected output
        assert torch.allclose(
            normalized_x[batch_indices],
            expected_output), f"Normalization failed for batch {i}"


@pytest.mark.parametrize('embed_dim', [16, 32, 64])
@pytest.mark.parametrize('dropout_rate', [0.0, 0.2, 0.4])
@pytest.mark.parametrize('last_act', [False, True])
def test_geognnconv(embed_dim, dropout_rate, last_act):
    conv = GeoGNNConv(embed_dim, dropout_rate, last_act)

    assert conv.embed_dim == embed_dim
    assert conv.last_act == last_act

    assert isinstance(conv.geo_conv, ModGINConv)
    assert isinstance(conv.graph_norm, GraphNorm)
    assert conv.act is None if not last_act else True
    assert isinstance(conv.dropout, torch.nn.Dropout)

    x = torch.randn(10, embed_dim)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.long)
    edge_attr = torch.randn(10, embed_dim)
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)

    out = conv(x, edge_index, edge_attr, batch=batch)

    assert out.size() == (10, embed_dim)


@pytest.mark.parametrize('in_channels, out_channels', [(16, 32), (32, 64),
                                                       (64, 128)])
def test_modginconv(in_channels, out_channels):
    conv = ModGINConv(in_channels, out_channels)

    assert isinstance(conv.lin, torch.nn.Linear)
    assert isinstance(conv.edge_lin, torch.nn.Linear)

    x = torch.randn(10, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.long)
    edge_attr = torch.randn(10, in_channels)

    out = conv(x, edge_index, edge_attr)

    assert out.size() == (10, out_channels)

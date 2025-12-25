import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.nn.models import TensorGNAN


def _dummy_data(num_nodes: int = 5, num_feats: int = 4):
    x = torch.randn(num_nodes, num_feats)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    # full distance matrix:
    rand_dist = torch.rand(num_nodes, num_nodes)
    rand_dist = (rand_dist + rand_dist.t()) / 2  # symmetric
    # Use a simple normalisation matrix of ones to avoid division issues
    norm = torch.ones_like(rand_dist)
    data = Data(x=x, edge_index=edge_index)
    data.node_distances = rand_dist
    data.normalization_matrix = norm
    return data


def test_tensor_gnan_graph_level():
    data = _dummy_data()
    model = TensorGNAN(in_channels=data.num_features, out_channels=2,
                       n_layers=2, hidden_channels=8)
    out = model(data)  # [1, 2]
    assert out.shape == (1, 2)


def test_tensor_gnan_feature_groups():
    """Ensure model works correctly with custom feature grouping."""
    data = _dummy_data(num_nodes=6, num_feats=4)

    # Group features [0,1] together; keep [2] and [3] separate.
    feature_groups = [[0, 1], [2], [3]]

    model = TensorGNAN(
        in_channels=data.num_features,
        out_channels=3,
        n_layers=1,  # single Linear layer per MLP for easier inspection
        feature_groups=feature_groups,
        normalize_rho=False,  # avoid dependence on normalisation matrix shape
    )

    out = model(data)  # [1, 3]

    # Forward pass shape check
    assert out.shape == (1, 3)

    # There should be exactly len(feature_groups) MLPs
    assert len(model.fs) == len(feature_groups)

    # The first MLP should accept 2 input features (because group size == 2)
    first_mlp = model.fs[0]
    assert isinstance(first_mlp.net, torch.nn.Linear)
    assert first_mlp.net.in_features == 2


def test_tensor_gnan_node_importance():
    """Node contributions should sum to graph‐level prediction."""
    data = _dummy_data(num_nodes=5, num_feats=3)

    model = TensorGNAN(
        in_channels=data.num_features,
        out_channels=4,
        n_layers=1,
        normalize_rho=False,  # simplifies the equality check
    )

    graph_out = model(data)  # [1, 4]
    node_contrib = model.node_importance(data)  # [N, 4]

    # Shape checks
    assert node_contrib.shape == (data.num_nodes, 4)

    # Sum of node contributions equals the graph‐level prediction (Eq. 3)
    contrib_sum = node_contrib.sum(dim=0, keepdim=True)  # [1, 4]
    assert torch.allclose(contrib_sum, graph_out, atol=1e-5)


def test_tensor_gnan_multiple_layers():
    """Model runs with multiple layers in the MLPs."""
    data = _dummy_data(num_nodes=7, num_feats=3)

    model = TensorGNAN(
        in_channels=data.num_features,
        out_channels=5,
        n_layers=3,
        hidden_channels=8,
        dropout=0.0,
    )

    out = model(data)
    assert out.shape == (1, 5)


def test_tensor_gnan_batched_data():
    """Batched graphs should be processed independently and aggregated
    per-graph.
    """
    g1 = _dummy_data(num_nodes=3, num_feats=4)
    g2 = _dummy_data(num_nodes=4, num_feats=4)

    # Build a single Data with block-diagonal distances and norms and a
    # batch vector
    x = torch.cat([g1.x, g2.x], dim=0)
    dist = torch.block_diag(g1.node_distances, g2.node_distances)
    norm = torch.block_diag(g1.normalization_matrix, g2.normalization_matrix)
    batch = torch.tensor([0] * g1.num_nodes + [1] * g2.num_nodes)

    batched = Data(x=x)
    batched.node_distances = dist
    batched.normalization_matrix = norm
    batched.batch = batch

    model = TensorGNAN(
        in_channels=4,
        out_channels=3,
        n_layers=2,
        hidden_channels=6,
        dropout=0.0,
        graph_level=True,
    )
    model.eval()

    out_batched = model(batched)  # [2, 3]
    assert out_batched.shape == (2, 3)

    # Compare against processing each graph separately with the same model
    out_g1 = model(g1)  # [1, 3]
    out_g2 = model(g2)  # [1, 3]
    stacked = torch.cat([out_g1, out_g2], dim=0)

    assert torch.allclose(out_batched, stacked, atol=1e-5)


def test_tensor_gnan_invalid_feature_groups_empty():
    data = _dummy_data(num_nodes=5, num_feats=3)
    with pytest.raises(ValueError, match="cannot be empty"):
        TensorGNAN(
            in_channels=data.num_features,
            out_channels=2,
            n_layers=1,
            feature_groups=[[0], [], [2]],
        )


def test_tensor_gnan_invalid_feature_groups_duplicate():
    data = _dummy_data(num_nodes=5, num_feats=3)
    with pytest.raises(ValueError, match="appears in multiple groups"):
        TensorGNAN(
            in_channels=data.num_features,
            out_channels=2,
            n_layers=1,
            feature_groups=[[0, 1], [1], [2]],
        )

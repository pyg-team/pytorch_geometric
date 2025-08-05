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


# -----------------------------------------------------------------------------
# New tests for feature grouping and node importance
# -----------------------------------------------------------------------------


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

    graph_out = model(data)            # [1, 4]
    node_contrib = model.node_importance(data)  # [N, 4]

    # Shape checks
    assert node_contrib.shape == (data.num_nodes, 4)

    # Sum of node contributions equals the graph‐level prediction (Eq. 3)
    contrib_sum = node_contrib.sum(dim=0, keepdim=True)  # [1, 4]
    assert torch.allclose(contrib_sum, graph_out, atol=1e-5)

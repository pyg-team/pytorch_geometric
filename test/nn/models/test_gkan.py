import torch

from torch_geometric.nn.models.gkan import GKANModel


def test_gkan() -> None:
    num_nodes = 10
    num_features = 128
    num_classes = 3
    num_layers = 3
    num = 5  # Number of grid intervals
    k = 3  # Polynomial order of splines
    architecture = 2  # KAN -> Aggregate architecture

    # Mock input data
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ], dtype=torch.long)

    # Initialize GKANModel
    model = GKANModel(
        input_dim=num_features,
        hidden_dim=64,
        output_dim=num_classes,
        num_layers=num_layers,
        num=num,  # Spline grid intervals
        k=k,  # Spline polynomial order
        architecture=architecture)

    # Check model string representation
    assert str(model).startswith('GKANModel')

    # Check forward pass in training mode
    model.train()
    out = model(x, edge_index)
    assert out.size() == (num_nodes, num_classes)
    assert out.min().item() >= -10 and out.max().item() <= 10

    # Check forward pass in evaluation mode
    model.eval()
    out = model(x, edge_index)
    assert out.size() == (num_nodes, num_classes)
    assert out.min().item() >= -5 and out.max().item() <= 5

    print("GKANModel test passed successfully!")


if __name__ == "__main__":
    test_gkan()

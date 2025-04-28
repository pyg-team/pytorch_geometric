import torch

from torch_geometric.nn.models.diffusion_gcn import DiffusionGCN


def test_diffusion_gcn_initialization():
    """Tests whether the DiffusionGCN model initializes correctly
    with the given parameters.
    """
    in_channels = 3
    out_channels = 8
    orders = 2

    # Initialize the model
    model = DiffusionGCN(in_channels, out_channels, orders)

    # Check the initialized parameters
    assert model.orders == orders, (
        "Orders parameter not initialized correctly.")
    assert model.num_matrices == 2 * orders + 1, (
        "Number of diffusion matrices not calculated correctly.")
    assert model.Theta1.shape == (
        in_channels * model.num_matrices,
        out_channels,
    ), "Theta1 weight matrix shape is incorrect."
    assert model.bias.shape == (out_channels, ), "Bias shape is incorrect."

    print("Initialization test passed.")


def test_diffusion_gcn_forward():
    """Unit test for the forward method of DiffusionGCN.
    Verifies that the output shape and values are valid.
    """
    # Test parameters
    batch_size = 4
    num_nodes = 10
    num_timesteps = 6
    in_channels = 6
    out_channels = 8
    orders = 2

    # Generate random input data
    X = torch.rand(batch_size, num_nodes, num_timesteps)
    A_q = torch.rand(num_nodes, num_nodes)  # Forward random walk matrix
    A_h = torch.rand(num_nodes, num_nodes)  # Backward random walk matrix

    # Initialize the model
    model = DiffusionGCN(in_channels, out_channels, orders)

    # Perform forward pass
    output = model(X, A_q, A_h)

    # Assert the output shape
    assert output.shape == (batch_size, num_nodes, out_channels), (
        f"Expected output shape {(batch_size, num_nodes, out_channels)}, "
        f"but got {output.shape}")

    # Check the output values are finite
    assert torch.isfinite(output).all(), "Output contains non-finite values."

    print("Forward method test passed.")


if __name__ == "__main__":
    # Run the tests
    test_diffusion_gcn_initialization()
    test_diffusion_gcn_forward()

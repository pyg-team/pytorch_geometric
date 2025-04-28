import torch

from torch_geometric.nn.models.bidirectional_tcn import BTCN


def test_bidirectional_tcn_forward():
    """Test the forward method of the BTCN class to ensure the output has
    the correct shape and is finite.
    """
    # Test parameters
    batch_size = 4
    num_timesteps = 10
    num_nodes = 5
    in_channels = 1  # Features per node
    out_channels = 8  # Desired output features
    kernel_size = 3

    # Generate random input data
    # X = torch.rand(batch_size, in_channels, num_nodes, num_timesteps)
    X = torch.rand(batch_size, num_timesteps, num_nodes)
    # Initialize the model
    model = BTCN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        activation="relu",
        device="cpu",
    )

    # Forward pass
    output = model(X)

    # Verify output shape
    assert output.shape == (batch_size, num_timesteps, out_channels), (
        f"Expected output shape {(batch_size, num_timesteps, out_channels)}, "
        f"but got {output.shape}")

    # Check that all values in the output are finite
    assert torch.isfinite(output).all(), "Output contains non-finite values."

    print("BTCN forward test passed.")


if __name__ == "__main__":
    # Run the test
    test_bidirectional_tcn_forward()

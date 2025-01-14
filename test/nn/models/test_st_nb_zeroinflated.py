import torch

from torch_geometric.nn.models.st_nb_zeroinflated import ST_NB_ZeroInflated


def test_st_nb_zero_inflated():
    """Test the forward pass of the ST_NB_ZeroInflated model."""
    # Test parameters
    batch_size = 4
    num_timesteps_input = 10
    num_timesteps_output = 8
    num_nodes = 5
    hidden_dim_s = 16
    rank_s = 8
    hidden_dim_t = 16
    rank_t = 8
    space_dim = num_nodes

    # Initialize the model
    model = ST_NB_ZeroInflated(
        num_timesteps_input=num_timesteps_input,
        hidden_dim_s=hidden_dim_s,
        rank_s=rank_s,
        num_timesteps_output=num_timesteps_output,
        space_dim=space_dim,
        hidden_dim_t=hidden_dim_t,
        rank_t=rank_t,
    )

    # Input data
    X = torch.rand(batch_size, num_timesteps_input, num_nodes)
    A_q = torch.eye(num_nodes)  # Forward random walk matrix
    A_h = torch.eye(num_nodes)  # Backward random walk matrix

    # Forward pass
    n_res, p_res, pi_res = model(X, A_q, A_h)

    # Verify output shapes
    expected_shape = (batch_size, num_timesteps_output, num_nodes)

    assert n_res.shape == expected_shape, (
        f"Expected n_res shape {expected_shape}, "
        f"but got {n_res.shape}")

    assert p_res.shape == expected_shape, (
        f"Expected p_res shape {expected_shape}, "
        f"but got {p_res.shape}")

    assert pi_res.shape == expected_shape, (
        f"Expected pi_res shape {expected_shape}, "
        f"but got {pi_res.shape}")

    # Check for finite values
    assert torch.isfinite(n_res).all(), "n_res contains non-finite values."
    assert torch.isfinite(p_res).all(), "p_res contains non-finite values."
    assert torch.isfinite(pi_res).all(), "pi_res contains non-finite values."

    print("ST_NB_ZeroInflated forward test passed.")


if __name__ == "__main__":
    # Run the test
    test_st_nb_zero_inflated()

import torch

from torch_geometric.nn.losses import ZINBLoss


def test_zinb_loss():
    # Create dummy data
    target = torch.tensor([0.0, 1.0, 2.0, 0.0, 4.0], dtype=torch.float32)
    mu = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0],
                      dtype=torch.float32)  # Mean of NB
    theta = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0],
                         dtype=torch.float32)  # Dispersion
    pi = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1],
                      dtype=torch.float32)  # Zero-inflation prob

    # Initialize the loss
    loss_fn = ZINBLoss()

    # Compute the loss
    loss = loss_fn((mu, theta, pi), target)

    # Check the loss value
    assert loss >= 0, "Loss should be non-negative."
    assert torch.isfinite(
        loss), "Loss should not contain NaN or infinite values."

    # Print the loss value for debugging
    print(f"ZINB Loss: {loss.item()}")


def test_zinb_loss_shapes():
    # Test with batched data
    batch_size = 4
    target = torch.rand(batch_size, 10)  # 10 targets per batch
    mu = torch.rand(batch_size, 10) + 0.1  # Mean of NB, avoiding zero
    theta = torch.rand(batch_size, 10) + 0.1  # Dispersion, avoiding zero
    pi = torch.rand(batch_size, 10)  # Zero-inflation probability

    # Initialize the loss
    loss_fn = ZINBLoss()

    # Compute the loss
    loss = loss_fn((mu, theta, pi), target)

    # Check the loss value
    assert loss >= 0, "Loss should be non-negative."
    assert torch.isfinite(
        loss), "Loss should not contain NaN or infinite values."

    # Print the loss value for debugging
    print(f"Batched ZINB Loss: {loss.item()}")


if __name__ == "__main__":
    test_zinb_loss()
    test_zinb_loss_shapes()

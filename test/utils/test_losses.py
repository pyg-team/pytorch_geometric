import torch

from torch_geometric.utils.losses import nb_zeroinflated_nll_loss


def test_nb_zeroinflated_nll_loss():
    """Test the nb_zeroinflated_nll_loss function."""
    # Test inputs
    y = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0])
    n = torch.tensor([1.0, 1.5, 2.0, 1.0, 1.8])
    p = torch.tensor([0.6, 0.7, 0.8, 0.5, 0.9])
    pi = torch.tensor([0.3, 0.4, 0.5, 0.2, 0.1])

    # Compute loss
    loss = nb_zeroinflated_nll_loss(y, n, p, pi)

    # Check loss type and value
    assert isinstance(loss,
                      torch.Tensor), ("Loss should return a torch.Tensor.")
    assert loss.item() > 0, "Loss value should be positive."

    print("Test passed: nb_zeroinflated_nll_loss works as expected.")


if __name__ == "__main__":
    test_nb_zeroinflated_nll_loss()

import torch

from torch_geometric.data import Data
from torch_geometric.nn.models.nbnorm_zeroinflated import NBNormZeroInflated


def test_forward_method():
    """Unit test for the forward method of NBNormZeroInflated.
    Checks that the outputs are of the correct shape and range.
    """
    # Create synthetic graph data
    num_nodes = 5
    input_features = 16
    output_features = 8
    edge_index = torch.tensor([[0, 1, 2, 3, 4],
                               [1, 2, 3, 4,
                                0]])  # 5 nodes connected in a cycle

    x = torch.rand(
        (num_nodes, input_features))  # Node features: [num_nodes, c_in]

    # Wrap data in PyG Data object
    data = Data(x=x, edge_index=edge_index)

    # Initialize the model
    model = NBNormZeroInflated(c_in=input_features, c_out=output_features)

    # Forward pass
    n, p, pi = model(data.x, data.edge_index)

    # Assert that the output shapes are correct
    assert n.shape == (
        num_nodes, output_features
    ), f"Expected n shape {(num_nodes, output_features)}, got {n.shape}"
    assert p.shape == (
        num_nodes, output_features
    ), f"Expected p shape {(num_nodes, output_features)}, got {p.shape}"
    assert pi.shape == (
        num_nodes, output_features
    ), f"Expected pi shape {(num_nodes, output_features)}, got {pi.shape}"

    # Assert that n is positive
    assert torch.all(n >= 0), "Parameter n contains negative values!"

    # Assert that p and pi are in the range [0, 1]
    assert torch.all((p >= 0)
                     & (p <= 1)), "Parameter p contains values outside [0, 1]!"
    assert torch.all((pi >= 0) &
                     (pi <= 1)), "Parameter pi contains values outside [0, 1]!"

    print("All tests passed for the forward method.")


if __name__ == "__main__":
    test_forward_method()

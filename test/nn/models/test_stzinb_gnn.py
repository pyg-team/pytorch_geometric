import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import STZINBGNN


def test_stzinb_gnn():
    # Define model parameters
    num_nodes = 50
    num_features = 10
    time_window = 5
    hidden_dim_s = 70
    hidden_dim_t = 7
    rank_s = 20
    rank_t = 4
    k = 4
    batch_size = 8

    # Create dummy data for testing
    edge_index = torch.randint(0, num_nodes, (2, 200),
                               dtype=torch.long)  # Random edges
    x = torch.rand(num_nodes, num_features)  # Random node features
    y = torch.randint(0, 10, (num_nodes, k))  # Random target for ZINB loss

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # Create a batch of graphs
    data_list = [data.clone() for _ in range(batch_size)]
    batch = Batch.from_data_list(data_list)  # Use Batch to create a batch

    # Initialize the model
    model = STZINBGNN(
        num_nodes=num_nodes,
        num_features=num_features,
        time_window=time_window,
        hidden_dim_s=hidden_dim_s,
        hidden_dim_t=hidden_dim_t,
        rank_s=rank_s,
        rank_t=rank_t,
        k=k,
    )

    # Forward pass
    pi, n, p = model(batch.x, batch.edge_index, batch.batch)

    # Assertions for output shapes
    assert pi.shape == (batch_size * num_nodes,
                        1), "Incorrect shape for π (zero-inflation parameter)"
    assert n.shape == (batch_size * num_nodes,
                       1), "Incorrect shape for n (shape parameter)"
    assert p.shape == (batch_size * num_nodes,
                       1), "Incorrect shape for p (probability parameter)"

    # Check value ranges
    assert (pi >= 0).all() and (pi <= 1).all(), "π values out of range [0, 1]"
    assert (n > 0).all(), "n values should be strictly positive"
    assert (p >= 0).all() and (p <= 1).all(), "p values out of range [0, 1]"

    print("STZINBGNN test passed successfully!")


# Run the test
if __name__ == "__main__":
    test_stzinb_gnn()

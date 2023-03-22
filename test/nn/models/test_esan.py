import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import DSnetwork, DSSnetwork, subgraph_pool


def test_subgraph_pool():
    # Define input data
    h_node = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    num_subgraphs = torch.tensor([2, 1])
    batch = torch.tensor([0, 0, 1])
    subgraph_batch = torch.tensor([0, 1, 0])

    # Call the function under test
    result = subgraph_pool(h_node,
                           batched_data(num_subgraphs, batch, subgraph_batch),
                           torch.mean)

    # Define expected output
    expected_output = torch.tensor([[2.0, 3.0], [2.0, 3.0], [5.0, 6.0]])

    # Compare the actual and expected output
    assert torch.allclose(result, expected_output)


# Define a helper function to create batched data
def batched_data(num_subgraphs, batch, subgraph_batch):
    class BatchedData:
        def __init__(self, num_subgraphs, batch, subgraph_batch):
            self.num_subgraphs = num_subgraphs
            self.batch = batch
            self.subgraph_batch = subgraph_batch

    return BatchedData(num_subgraphs, batch, subgraph_batch)


from typing import List, Optional

from my_module import DSnetwork, DSSnetwork


def batched_data():
    # Define example input batch data
    x = torch.randn(50, 10)  # node features
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 2,
                                              3]])  # graph connectivity
    subgraph_id_batch = torch.tensor([0, 0, 0, 1])  # subgraph IDs
    return torch_geometric.data.Batch(x=x, edge_index=edge_index,
                                      subgraph_id_batch=subgraph_id_batch)


def test_dsnetwork(batched_data):
    # Define example input
    fc_channels = [32, 64]
    num_tasks = 3
    subgraph_gnn = torch.nn.Sequential(
        torch_geometric.nn.MessagePassing(torch_geometric.nn.naive_relu,
                                          torch_geometric.nn.naive_add),
        torch.nn.Linear(16, 32), torch.nn.ReLU(),
        torch_geometric.nn.MessagePassing(torch_geometric.nn.naive_relu,
                                          torch_geometric.nn.naive_add),
        torch.nn.Linear(32, 64), torch.nn.ReLU())
    invariant = False

    # Initialize the DSnetwork object and test forward pass
    model = DSnetwork(fc_channels, num_tasks, subgraph_gnn, invariant)
    output = model(batched_data)
    assert output.shape == (batched_data.num_graphs, num_tasks)


def test_dssnetwork(batched_data):
    # Define example input
    num_layers = 3
    in_dim = 10
    emb_dim = 16
    num_tasks = 3
    feature_encoder = torch.nn.Sequential(torch.nn.Linear(10, 32),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(32, 16))
    GNNConv = torch_geometric.nn.GCNConv

    # Initialize the DSSnetwork object and test forward pass
    model = DSSnetwork(num_layers, in_dim, emb_dim, num_tasks, feature_encoder,
                       GNNConv)
    output = model(batched_data)
    assert output.shape == (batched_data.num_graphs, num_tasks)

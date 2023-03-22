import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import DSnetwork, DSSnetwork, subgraph_pool


def test_dsnetwork():
    pass


def test_dssnetwork():
    # Define the input graph
    x = torch.randn(4, 16)  # feature matrix for each node
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # edge connectivity matrix
    edge_attr = torch.randn(3, 8)  # edge feature matrix
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # batch assignment for each node

    # Create the DSSnetwork model
    num_layers = 2
    in_dim = 16
    emb_dim = 32
    num_tasks = 2
    feature_encoder = torch.nn.Linear(in_features=16, out_features=emb_dim)
    GNNConv = torch_geometric.nn.GCNConv
    model = DSSnetwork(num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv)

    # Compute the output of the model
    output = model(torch_geometric.data.Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch))

    # Check that the output has the correct shape
    assert output.shape == (2, num_tasks)

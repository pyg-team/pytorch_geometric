import pytest
import torch

from torch_geometric.data import Batch
from torch_geometric.nn.models import DSnetwork, DSSnetwork
from torch_geometric.nn import GINConv



def test_dsnetwork():
    pass



class GINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(GINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.layer = GINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


def test_dssnetwork():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 3)
    batch = torch.tensor([0, 0, 0, 0])
    num_nodes_per_subgraph = torch.tensor([2, 2])

    # Create a Batch object from the input data
    batched_data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, num_nodes_per_subgraph=num_nodes_per_subgraph)

    # Define model hyperparameters
    num_layers = 2
    in_dim = 8
    emb_dim = 16
    num_tasks = 1

    feature_encoder = torch.nn.Identity()
    GNNConv = GINConv

    model = DSSnetwork(num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv)

    output = model(batched_data)

    assert output.shape == (2, 1)

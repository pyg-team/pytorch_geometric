from itertools import product

import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphConv
from torch_geometric.nn.models import DSnetwork, DSSnetwork
from torch_geometric.transforms import subgraph_policy

n_subgraphs = [2, 5, 10]
num_tasks = [1, 2, 3]
policies = ['node_deletion', 'edge_deletion', 'ego', 'ego_plus']


@pytest.mark.parametrize('n_subgraphs, num_tasks, policy',
                         product(n_subgraphs, num_tasks, policies))
def test_dsnetwork(n_subgraphs, num_tasks, policy):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    # Create a Batch object from the input data
    data = Data(x=x, edge_index=edge_index)
    transform = subgraph_policy(policy, num_hops=1)
    subgraphs = transform(data)

    ls_subgraphs = [subgraphs] * n_subgraphs
    batched_data = Batch.from_data_list(ls_subgraphs,
                                        follow_batch=['subgraph_id'])

    # Define model hyperparameters
    num_layers = 2
    in_dim = 8 if policy != 'ego_plus' else 10
    emb_dim = 16
    GNNConv = GraphConv

    def feature_encoder(x):
        return x

    model = DSnetwork(num_layers, in_dim, emb_dim, num_tasks, feature_encoder,
                      GNNConv)

    output = model(batched_data)

    assert output.shape == (n_subgraphs, num_tasks)


@pytest.mark.parametrize('n_subgraphs, num_tasks, policy',
                         product(n_subgraphs, num_tasks, policies))
def test_dssnetwork(n_subgraphs, num_tasks, policy):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    # Create a Batch object from the input data
    data = Data(x=x, edge_index=edge_index)
    transform = subgraph_policy(policy, num_hops=1)
    subgraphs = transform(data)

    ls_subgraphs = [subgraphs] * n_subgraphs
    batched_data = Batch.from_data_list(ls_subgraphs,
                                        follow_batch=['subgraph_id'])

    # Define model hyperparameters
    num_layers = 2
    in_dim = 8 if policy != 'ego_plus' else 10
    emb_dim = 16
    GNNConv = GraphConv

    def feature_encoder(x):
        return x

    model = DSSnetwork(num_layers, in_dim, emb_dim, num_tasks, feature_encoder,
                       GNNConv)

    output = model(batched_data)

    assert output.shape == (n_subgraphs, num_tasks)

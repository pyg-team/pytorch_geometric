import torch

from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.datasets import MovieLens1B
from torch_geometric.testing import withPackage
from torch_geometric.testing import onlyFullTest

# @onlyFullTest
def test_movielens_1b():
    dataset = MovieLens1B(root="./data/MovieLens1B")
    data = dataset[0]

    assert str(dataset) == f'MovieLens1B()'

    assert isinstance(data, HeteroData)
    assert data.num_nodes == 3065854

    assert data['user'].num_nodes == 2210078
    assert data['movie'].num_nodes == 855776
    assert data['user', 'rates', 'movie'].num_edges == 1223962043

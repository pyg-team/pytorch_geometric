import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import AddPositionalEncoding


@pytest.mark.parametrize('name',
                         ['laplacian_eigenvector_pe', 'random_walk_pe'])
def test_add_positional_encoding(name):
    x = torch.tensor([0., 1., 2., 3., 4., 5.]).view(-1, 1)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    N, F = x.size()
    num_pe = 3

    transform_attr = AddPositionalEncoding(name, num_pe, 'attr')
    assert transform_attr.__repr__() == f'AddPositionalEncoding(name={name})'

    data = Data(x=x, edge_index=edge_index)
    out = transform_attr(data)
    assert hasattr(out, name)
    assert getattr(out, name).size() == (N, num_pe)

    transform_cat = AddPositionalEncoding(name, num_pe, 'cat')
    assert transform_attr.__repr__() == f'AddPositionalEncoding(name={name})'

    data = Data(x=x, edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe + F)

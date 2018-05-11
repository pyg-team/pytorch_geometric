import torch
from torch_geometric.data import Data


def test_data():
    x = torch.tensor([[1, 3, 5], [2, 4, 6]]).t()
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index)

    assert data.x.tolist() == x.tolist()
    assert data['x'].tolist() == x.tolist()

    assert data.keys() == ['x', 'edge_index']
    assert len(data) == 2
    assert 'x' in data and 'edge_index' in data and 'pos' not in data

    assert data.num_nodes == 3

    data.to(torch.device('cpu'))
    assert not data.x.is_contiguous()
    data.contiguous()
    assert data.x.is_contiguous()

    data['x'] = x + 1
    assert data.x.tolist() == (x + 1).tolist()

    assert data.__repr__() == 'Data(x=[3, 2], edge_index=[2, 4])'

    dictionary = {'x': x, 'edge_index': edge_index}
    data = Data.from_dict(dictionary)
    assert data.keys() == ['x', 'edge_index']

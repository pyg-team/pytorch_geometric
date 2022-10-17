import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import Constant


def test_constant():
    assert str(Constant()) == 'Constant(value=1.0)'

    x = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]])

    data = Data(edge_index=edge_index, num_nodes=3)
    data = Constant()(data)
    assert len(data) == 3
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.x.tolist() == [[1], [1], [1]]
    assert data.num_nodes == 3

    data = Data(edge_index=edge_index, x=x)
    data = Constant()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.x.tolist() == [[-1, 0, 1], [0, 0, 1], [2, 0, 1]]

    data = HeteroData()
    data['v'].x = x
    data = Constant()(data)
    assert len(data) == 1
    assert data['v'].x.tolist() == [[-1, 0, 1], [0, 0, 1], [2, 0, 1]]

    data = HeteroData()
    data['v'].x = x
    data['w'].x = x
    data = Constant(node_types='w')(data)
    assert len(data) == 1
    assert data['v'].x.tolist() == x.tolist()
    assert data['w'].x.tolist() == [[-1, 0, 1], [0, 0, 1], [2, 0, 1]]

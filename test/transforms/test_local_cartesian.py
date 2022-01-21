import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LocalCartesian


def test_local_cartesian():
    assert LocalCartesian().__repr__() == 'LocalCartesian()'

    pos = torch.Tensor([[-1, 0], [0, 0], [2, 0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.Tensor([1, 2, 3, 4])

    data = Data(edge_index=edge_index, pos=pos)
    data = LocalCartesian()(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.edge_attr.tolist() == [[0.25, 0.5], [1.0, 0.5], [0.0, 0.5],
                                       [1.0, 0.5]]

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = LocalCartesian()(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.edge_attr.tolist() == [[1, 0.25, 0.5], [2, 1.0, 0.5],
                                       [3, 0.0, 0.5], [4, 1.0, 0.5]]

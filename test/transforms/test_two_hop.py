import torch
from torch_geometric.transforms import TwoHop
from torch_geometric.data import Data


def test_two_hop():
    assert TwoHop().__repr__() == 'TwoHop()'

    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.float)

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data = TwoHop()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]]
    assert data.edge_attr.tolist() == [1, 2, 3, 1, 0, 0, 2, 0, 0, 3, 0, 0]

    data = Data(edge_index=edge_index)
    data = TwoHop()(data)
    assert len(data) == 1
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]]

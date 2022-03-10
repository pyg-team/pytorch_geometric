import torch

from torch_geometric.data import Data
from torch_geometric.transforms import VirtualNode


def test_virtual_node():
    assert str(VirtualNode()) == 'VirtualNode()'

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[2, 0, 2], [3, 1, 0]])
    edge_weight = torch.rand(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=x.size(0))

    data = VirtualNode()(data)
    assert len(data) == 6

    assert data.x.size() == (5, 16)
    assert torch.allclose(data.x[:4], x)
    assert data.x[4:].abs().sum() == 0

    assert data.edge_index.tolist() == [[2, 0, 2, 0, 1, 2, 3, 4, 4, 4, 4],
                                        [3, 1, 0, 4, 4, 4, 4, 0, 1, 2, 3]]

    assert data.edge_weight.size() == (11, )
    assert torch.allclose(data.edge_weight[:3], edge_weight)
    assert data.edge_weight[3:].abs().sum() == 8

    assert data.edge_attr.size() == (11, 8)
    assert torch.allclose(data.edge_attr[:3], edge_attr)
    assert data.edge_attr[3:].abs().sum() == 0

    assert data.num_nodes == 5

    assert data.edge_type.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

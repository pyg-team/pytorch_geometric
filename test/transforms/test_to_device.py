import torch

from torch_geometric.data import Data
from torch_geometric.testing import withCUDA
from torch_geometric.transforms import ToDevice


@withCUDA
def test_to_device(device):
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    transform = ToDevice(device)
    assert str(transform) == f'ToDevice({device})'

    data = transform(data)
    for key, value in data:
        assert value.device == device

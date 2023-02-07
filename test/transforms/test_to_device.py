import torch

from torch_geometric.data import Data
from torch_geometric.transforms import ToDevice


def test_to_device():
    assert str(ToDevice('cpu')) == 'ToDevice(cpu)'

    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    if (torch.cuda.is_available()):
        to_device = ToDevice('cuda')
        data = to_device(data)
        assert all([val.is_cuda for _, val in data.stores[0].items()])

    to_device = ToDevice('cpu')
    assert all([val.is_cpu for _, val in data.stores[0].items()])

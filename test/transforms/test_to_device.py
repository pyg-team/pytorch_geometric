import torch

from torch_geometric.data import Data
from torch_geometric.transforms import ToDevice


def test_to_device():
    assert ToDevice(0).__repr__() == 'ToDevice(0)'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)

    if (torch.cuda.is_available()):
        to_device_obj = ToDevice(0)
        data = to_device_obj(data)
        assert to_device_obj.device == 0

    to_device_obj = ToDevice('cpu')
    data = to_device_obj(data)
    assert to_device_obj.device == 'cpu'

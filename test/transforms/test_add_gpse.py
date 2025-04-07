import os.path as osp

import torch

from torch_geometric.data import Data
from torch_geometric.nn import GPSE
from torch_geometric.transforms import AddGPSE

num_nodes = 6
gpse_inner_dim = 512


def test_gpse():
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    model = GPSE.from_pretrained(
        name='molpcba', root=osp.join(osp.dirname(osp.realpath(__file__)),
                                      '..', 'data', 'GPSE_pretrained'))
    transform = AddGPSE(model)
    assert str(transform) == 'AddGPSE()'
    out = transform(data)
    assert out.pestat_GPSE.size() == (num_nodes, gpse_inner_dim)

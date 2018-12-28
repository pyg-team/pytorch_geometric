import torch
from torch_geometric.utils import scatter_


def test_scatter_():
    index = torch.tensor([0, 0, 1, 1])
    src = torch.Tensor([-4, -2, 2, 4])

    assert scatter_('add', src, index).tolist() == [-6, 6]
    assert scatter_('mean', src, index).tolist() == [-3, 3]
    assert scatter_('max', src, index).tolist() == [-2, 4]

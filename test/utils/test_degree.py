import torch
from torch_geometric.utils import degree


def test_degree():
    row = torch.tensor([0, 1, 0, 2, 0])
    assert degree(row).tolist() == [3, 1, 1]

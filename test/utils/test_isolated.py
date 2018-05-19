import torch
from torch_geometric.utils import contains_isolated_nodes


def test_contains_isolated_nodes():
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])

    assert not contains_isolated_nodes(torch.stack([row, col], dim=0))
    assert contains_isolated_nodes(torch.stack([row, col], dim=0), 3)

    row = torch.tensor([0, 1, 2, 0])
    col = torch.tensor([1, 0, 2, 0])
    assert contains_isolated_nodes(torch.stack([row, col], dim=0))

import torch

from torch_geometric.utils import is_undirected, to_undirected


def test_is_undirected():
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])
    sym_weight = torch.tensor([0, 0, 1])
    asym_weight = torch.tensor([0, 1, 1])

    assert is_undirected(torch.stack([row, col], dim=0))
    assert is_undirected(torch.stack([row, col], dim=0), sym_weight)
    assert not is_undirected(torch.stack([row, col], dim=0), asym_weight)

    row = torch.tensor([0, 1, 1])
    col = torch.tensor([1, 0, 2])

    assert not is_undirected(torch.stack([row, col], dim=0))


def test_to_undirected():
    row = torch.tensor([0, 1, 1])
    col = torch.tensor([1, 0, 2])

    edge_index = to_undirected(torch.stack([row, col], dim=0))
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

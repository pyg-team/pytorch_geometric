import torch

from torch_geometric.nn import PositionalEncoding


def test_positional_encoding():
    encoder = PositionalEncoding(64)
    assert str(encoder) == 'PositionalEncoding(64)'

    x = torch.tensor([1.0, 2.0, 3.0])
    assert encoder(x).size() == (3, 64)

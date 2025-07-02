import torch

from torch_geometric.nn import PositionalEncoding, TemporalEncoding
from torch_geometric.testing import withDevice


@withDevice
def test_positional_encoding(device):
    encoder = PositionalEncoding(64, device=device)
    assert str(encoder) == 'PositionalEncoding(64)'

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert encoder(x).size() == (3, 64)


@withDevice
def test_temporal_encoding(device):
    encoder = TemporalEncoding(64, device=device)
    assert str(encoder) == 'TemporalEncoding(64)'

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert encoder(x).size() == (3, 64)

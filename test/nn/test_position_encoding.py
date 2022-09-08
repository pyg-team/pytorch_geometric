import pytest
import torch

from torch_geometric.nn.position_encoding import PositionEncoding1D


@pytest.mark.parametrize('mode', ['sin', 'learn'])
def test_position_encoding_1d(mode):
    pos_encoder = PositionEncoding1D(100, 512, mode=mode)
    assert str(pos_encoder) == f'PositionEncoding1D(100, 512, mode={mode})'
    pos_ids = torch.LongTensor([1, 2, 3])
    out = pos_encoder(pos_ids)
    assert out.size() == (3, 512)

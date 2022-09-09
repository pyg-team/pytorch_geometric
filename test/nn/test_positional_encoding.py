import pytest
import torch

from torch_geometric.nn.positional_encoding import PositionalEncoding1D


@pytest.mark.parametrize('mode_args', [('sin', None), ('learn', 100)])
def test_positional_encoding_1d(mode_args):
    mode, max_position = mode_args
    pos_encoder = PositionalEncoding1D(512, max_position, mode=mode)
    assert str(pos_encoder) == ('PositionalEncoding1D(512, '
                                f'max_position={max_position}, mode={mode})')
    position = torch.LongTensor([1, 2, 3])
    out = pos_encoder(position)
    assert out.size() == (3, 512)

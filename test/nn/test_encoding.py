import pytest
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


@withDevice
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_positional_encoding_preserves_input_dtype(device, dtype):
    # `frequency` is registered as a plain float32 buffer; the forward pass
    # must not silently upcast a lower-precision input to float32.
    encoder = PositionalEncoding(64, device=device)
    x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    assert encoder(x).dtype == dtype


@withDevice
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_temporal_encoding_preserves_input_dtype(device, dtype):
    # `weight` is registered as a plain float32 buffer; the forward pass
    # uses `@` (matmul), which raises instead of promoting on a dtype
    # mismatch, so an unfixed version crashes here rather than just
    # returning the wrong dtype.
    encoder = TemporalEncoding(64, device=device)
    x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    assert encoder(x).dtype == dtype

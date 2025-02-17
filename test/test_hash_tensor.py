import pytest
import torch

from torch_geometric import HashTensor
from torch_geometric.testing import has_package, withCUDA

KEY_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]


@withCUDA
@pytest.mark.skipif(
    not has_package('pyg-lib') and not has_package('pandas'),
    reason='Missing dependencies',
)
@pytest.mark.parametrize('dtype', KEY_DTYPES)
def test_basic(dtype, device):
    if dtype != torch.bool:
        key = torch.tensor([2, 1, 0], dtype=dtype, device=device)
    else:
        key = torch.tensor([True, False], device=device)
    value = torch.randn(key.size(0), 2, device=device)

    HashTensor(key, value)


@withCUDA
@pytest.mark.skipif(
    not has_package('pyg-lib') and not has_package('pandas'),
    reason='Missing dependencies',
)
def test_string_key(device):
    HashTensor(['1', '2', '3'], device=device)


@withCUDA
@pytest.mark.skipif(
    not has_package('pyg-lib') and not has_package('pandas'),
    reason='Missing dependencies',
)
def test_to_function(device):
    key = torch.tensor([2, 1, 0], device=device)
    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    out = tensor.to(device)
    assert isinstance(out, HashTensor)
    assert id(out) == id(tensor)
    assert out.device == device
    assert out._value.device == device
    assert out._min_key.device == device
    assert out._max_key.device == device

    out = tensor.to('cpu')
    assert isinstance(out, HashTensor)
    if key.is_cuda:
        assert id(out) != id(tensor)
    else:
        assert id(out) == id(tensor)
    assert out.device == torch.device('cpu')
    assert out._value.device == torch.device('cpu')
    assert out._min_key.device == torch.device('cpu')
    assert out._max_key.device == torch.device('cpu')

    out = tensor.double()
    assert isinstance(out, HashTensor)
    assert out._value.dtype == torch.double

import pytest
import torch

from torch_geometric import HashTensor
from torch_geometric.testing import withCUDA, withPackage

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
@withPackage('pyg_lib')  # TODO Remove
@pytest.mark.parametrize('dtype', KEY_DTYPES)
def test_basic(dtype, device):
    if dtype != torch.bool:
        key = torch.tensor([2, 1, 0], dtype=dtype, device=device)
    else:
        key = torch.tensor([True, False], device=device)
    value = torch.randn(key.size(0), 2, device=device)

    HashTensor(key, value)


@withCUDA
@withPackage('pyg_lib')  # TODO Remove
def test_string_key(device):
    HashTensor(['1', '2', '3'], device=device)

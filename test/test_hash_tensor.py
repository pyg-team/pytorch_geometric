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


@withCUDA
@pytest.mark.parametrize('dtype', KEY_DTYPES[:1])
def test_index_select(dtype, device):
    if dtype != torch.bool:
        key = torch.tensor([2, 1, 0], dtype=dtype, device=device)
    else:
        key = torch.tensor([True, False], device=device)
    value = torch.randn(key.size(0), 2, device=device)

    tensor = HashTensor(key, value)

    import numpy as np
    out = torch.index_select(tensor, 0, np.array([0, 1]))

    torch.index_select(tensor, 0, torch.tensor([0, 1]))
    out = torch.zeros_like(value)
    torch.index_select(tensor, 0, torch.tensor([0, 1]), out=out)
    # print(out)

    print('--------')
    print('...')
    out = tensor[...]  # done
    # print(out)
    print('..., [1, 0]')
    out = tensor[..., torch.tensor([1, 0])]  # done
    print('..., [1, 0], [1, 0]')
    out = tensor[..., torch.tensor([1, 0]), torch.tensor([1, 0])]  # done
    print('..., [1, 0], None')
    out = tensor[..., torch.tensor([1, 0]), None]  # done
    print('[1, 0], None')
    out = tensor[torch.tensor([1, 0]), None]  # done
    print('None, [1, 0]')
    out = tensor[None, torch.tensor([1, 0])]  # done
    # print(out)
    print('--------')
    # out = tensor[None, torch.tensor([1, 0])]  # done, not doable
    # print(out)
    # print('--------')
    # out = tensor[..., torch.tensor([1, 0]), torch.tensor([1])]  # index
    # print(out)
    # print('--------')
    # out = tensor[np.array(['1', '0']), torch.tensor([True, False])]  # index
    # print(out)
    # print('--------')

    # kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    # adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row', **kwargs)

    # index = tensor([1, 3], device=device)
    # out = adj.index_select(1, index)
    # assert out.equal(tensor([[1, 2], [0, 1]], device=device))
    # assert isinstance(out, EdgeIndex)
    # assert not out.is_sorted
    # assert not out.is_undirected

    # index = tensor([0], device=device)
    # out = adj.index_select(0, index)
    # assert out.equal(tensor([[0, 1, 1, 2]], device=device))
    # assert not isinstance(out, EdgeIndex)

    # index = tensor([1, 3], device=device)
    # inplace = torch.empty(2, 2, dtype=dtype, device=device)
    # out = torch.index_select(adj, 1, index, out=inplace)
    # assert out.data_ptr() == inplace.data_ptr()
    # assert not isinstance(out, EdgeIndex)
    # assert not isinstance(inplace, EdgeIndex)

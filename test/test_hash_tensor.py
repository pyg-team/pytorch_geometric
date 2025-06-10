import os.path as osp
from typing import List

import numpy as np
import pytest
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import HashTensor
from torch_geometric.io import fs
from torch_geometric.testing import (
    onlyCUDA,
    onlyLinux,
    withCUDA,
    withHashTensor,
    withPackage,
)

KEY_DTYPES = [
    pytest.param(torch.bool, id='bool'),
    pytest.param(torch.uint8, id='uint8'),
    pytest.param(torch.int8, id='int8'),
    pytest.param(torch.int16, id='int16'),
    pytest.param(torch.int32, id='int32'),
    pytest.param(torch.int64, id='int64'),
    pytest.param(torch.float16, id='float16'),
    pytest.param(torch.bfloat16, id='bfloat16'),
    pytest.param(torch.float32, id='float32'),
    pytest.param(torch.float64, id='float64'),
]


@withCUDA
@withHashTensor
@pytest.mark.parametrize('dtype', KEY_DTYPES)
def test_basic(dtype, device):
    if dtype != torch.bool:
        key = torch.tensor([2, 1, 0], dtype=dtype, device=device)
    else:
        key = torch.tensor([True, False], device=device)

    tensor = HashTensor(key)
    if tensor.is_cuda:
        assert str(tensor) == (f"HashTensor({tensor.as_tensor().tolist()}, "
                               f"device='{tensor.device}')")
    else:
        assert str(tensor) == f"HashTensor({tensor.as_tensor().tolist()})"

    assert tensor.dtype == torch.int64
    assert tensor.device == device
    assert tensor.size() == (key.size(0), )

    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)
    assert str(tensor).startswith("HashTensor([")
    assert tensor.dtype == torch.float
    assert tensor.device == device
    assert tensor.size() == (key.size(0), 2)


@withCUDA
@withHashTensor
@pytest.mark.parametrize('dtype', KEY_DTYPES)
def test_empty(dtype, device):
    key = torch.empty(0, dtype=dtype, device=device)
    tensor = HashTensor(key)
    assert tensor.dtype == torch.int64
    assert tensor.device == device
    assert tensor.size() == (0, )

    out = tensor.index_select(0, torch.empty(0, dtype=dtype, device=device))
    assert not isinstance(out, HashTensor)
    assert out.dtype == torch.int64
    assert out.device == device
    assert out.size() == (0, )

    value = torch.empty(0, device=device)
    tensor = HashTensor(key, value)
    assert tensor.dtype == value.dtype
    assert tensor.device == device
    assert tensor.size() == (0, )

    out = tensor.index_select(0, torch.empty(0, dtype=dtype, device=device))
    assert not isinstance(out, HashTensor)
    assert out.dtype == value.dtype
    assert out.device == device
    assert out.size() == (0, )


@withCUDA
@withHashTensor
def test_string_key(device):
    tensor = HashTensor(['1', '2', '3'], device=device)
    out = tensor[['3', '2', '4']]
    assert out.equal(torch.tensor([2, 1, -1], device=device))


@withCUDA
@withHashTensor
def test_clone(device):
    key = torch.tensor([2, 1, 0], device=device)
    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    out = tensor.clone()
    assert isinstance(out, HashTensor)
    assert out.dtype == tensor.dtype
    assert out.device == tensor.device
    assert out._value.data_ptr() != tensor._value.data_ptr()

    out = torch.clone(tensor)
    assert isinstance(out, HashTensor)
    assert out.dtype == tensor.dtype
    assert out.device == tensor.device
    assert out._value.data_ptr() != tensor._value.data_ptr()


@withCUDA
@withHashTensor
def test_share_memory(device):
    key = torch.tensor([2, 1, 0], device=device)
    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    out = tensor.share_memory_()
    assert isinstance(out, HashTensor)
    assert out.is_shared()
    assert out._value.is_shared()
    assert out.data_ptr() == tensor.data_ptr()


@onlyCUDA
@withHashTensor
def test_pin_memory():
    key = torch.tensor([2, 1, 0])
    value = torch.randn(key.size(0), 2)
    tensor = HashTensor(key, value)

    assert not tensor.is_pinned()
    out = tensor.pin_memory()
    assert isinstance(out, HashTensor)
    assert out.is_pinned()


@withCUDA
@withHashTensor
def test_detach(device):
    key = torch.tensor([2, 1, 0], device=device)
    value = torch.randn(key.size(0), 2, device=device, requires_grad=True)
    tensor = HashTensor(key, value)

    assert tensor.requires_grad
    out = tensor.detach()
    assert isinstance(out, HashTensor)
    assert not out.requires_grad
    assert not out._value.requires_grad

    tensor.detach_()
    assert not tensor.requires_grad
    assert not tensor._value.requires_grad


@withCUDA
@withHashTensor
def test_contiguous(device):
    key = torch.tensor([2, 1, 0], device=device)
    value = torch.randn(2, key.size(0), device=device).t()
    assert not value.is_contiguous()

    tensor = HashTensor(key, value)
    assert not tensor.is_contiguous()
    out = tensor.contiguous()
    assert out.is_contiguous()
    assert out._value.is_contiguous()


@withCUDA
@withHashTensor
def test_save_and_load(device, tmp_path):
    key = torch.tensor([2, 1, 0], device=device)
    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    path = osp.join(tmp_path, 'hash_tensor.pt')
    torch.save(tensor, path)
    out = fs.torch_load(path)

    assert isinstance(out, HashTensor)
    assert out._value.equal(value)
    assert out._min_key.equal(key.min())
    assert out._max_key.equal(key.max())


@withCUDA
@withHashTensor
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
@withHashTensor
def test_unsqueeze(device):
    key = torch.tensor([2, 1, 0], device=device)
    tensor = HashTensor(key)

    with pytest.raises(IndexError, match="in the first dimension"):
        tensor.unsqueeze(0)

    with pytest.raises(IndexError, match="in the first dimension"):
        tensor.unsqueeze(-2)

    with pytest.raises(IndexError, match="out of range"):
        tensor.unsqueeze(2)

    with pytest.raises(IndexError, match="out of range"):
        tensor.unsqueeze(-3)

    out = tensor.unsqueeze(-1)
    assert out.size() == (3, 1)
    assert out._value is not None

    out = tensor[..., None]
    assert out.size() == (3, 1)
    assert out._value is not None

    out = tensor[..., None, None]
    assert out.size() == (3, 1, 1)
    assert out._value is not None

    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    out = tensor.unsqueeze(-1)
    assert out.size() == (3, 2, 1)
    assert out._value is not None

    out = tensor[..., None]
    assert out.size() == (3, 2, 1)
    assert out._value is not None

    out = tensor[..., None, None]
    assert out.size() == (3, 2, 1, 1)
    assert out._value is not None

    out = tensor.unsqueeze(1)
    assert out.size() == (3, 1, 2)
    assert out._value is not None


@withCUDA
@withHashTensor
@pytest.mark.parametrize('num_keys', [3, 1])
def test_squeeze(num_keys, device):
    key = torch.tensor([2, 1, 0][:num_keys], device=device)
    tensor = HashTensor(key)

    out = tensor.squeeze()
    assert isinstance(out, HashTensor)
    assert out.size() == (num_keys, )

    out = tensor.squeeze(0)
    assert isinstance(out, HashTensor)
    assert out.size() == (num_keys, )

    out = tensor.squeeze(-1)
    assert isinstance(out, HashTensor)
    assert out.size() == (num_keys, )

    if torch_geometric.typing.WITH_PT20:
        out = tensor.squeeze([0])
        assert isinstance(out, HashTensor)
        assert out.size() == (num_keys, )

    with pytest.raises(IndexError, match="out of range"):
        tensor.squeeze(1)

    with pytest.raises(IndexError, match="out of range"):
        tensor.squeeze(-2)

    value = torch.randn(key.size(0), 1, 1, device=device)
    tensor = HashTensor(key, value)

    out = tensor.squeeze()
    assert isinstance(out, HashTensor)
    assert out.size() == (num_keys, )

    out = tensor.squeeze(0)
    assert isinstance(out, HashTensor)
    assert out.size() == (num_keys, 1, 1)

    out = tensor.squeeze(-1)
    assert isinstance(out, HashTensor)
    assert out.size() == (num_keys, 1)

    if torch_geometric.typing.WITH_PT20:
        out = tensor.squeeze([0, 1, 2])
        assert isinstance(out, HashTensor)
        assert out.size() == (num_keys, )


@withCUDA
@withHashTensor
def test_slice(device):
    key = torch.tensor([2, 1, 0], device=device)
    tensor = HashTensor(key)

    with pytest.raises(IndexError, match="out of range"):
        torch.narrow(tensor, dim=-2, start=0, length=2)

    out = tensor[:]
    assert isinstance(out, HashTensor)
    assert out._value is None

    out = tensor[-2:4]
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(torch.tensor([1, 2], device=device))

    out = tensor[..., 0:2]
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(torch.tensor([0, 1], device=device))

    out = torch.narrow(tensor, dim=0, start=2, length=1)
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(torch.tensor([2], device=device))

    out = tensor.narrow(dim=0, start=1, length=2)
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(torch.tensor([1, 2], device=device))

    value = torch.randn(key.size(0), 4, device=device)
    tensor = HashTensor(key, value)

    out = tensor[0:2]
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(value[0:2])

    out = tensor[..., 0:2]
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(value[..., 0:2])

    out = torch.narrow(tensor, dim=1, start=2, length=1)
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(value[..., 2:3])


@withCUDA
@withHashTensor
@pytest.mark.parametrize('dtype', KEY_DTYPES)
def test_index_select(dtype, device):
    if dtype != torch.bool:
        key = torch.tensor([2, 1, 0], dtype=dtype, device=device)
        query = torch.tensor([0, 3, 2], dtype=dtype, device=device)
    else:
        key = torch.tensor([True, False], device=device)
        query = torch.tensor([False, True], device=device)

    tensor = HashTensor(key)

    out = torch.index_select(tensor, 0, query)
    assert not isinstance(out, HashTensor)
    if dtype != torch.bool:
        assert out.equal(torch.tensor([2, -1, 0], device=device))
    else:
        assert out.equal(torch.tensor([1, 0], device=device))

    out = tensor.index_select(-1, query)
    assert not isinstance(out, HashTensor)
    if dtype != torch.bool:
        assert out.equal(torch.tensor([2, -1, 0], device=device))
    else:
        assert out.equal(torch.tensor([1, 0], device=device))

    with pytest.raises(IndexError, match="out of range"):
        torch.index_select(tensor, 1, query)

    with pytest.raises(IndexError, match="out of range"):
        tensor.index_select(-2, query)

    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    out = torch.index_select(tensor, 0, query)
    assert not isinstance(out, HashTensor)
    if dtype != torch.bool:
        expected = torch.full_like(value, float('NaN'))
        expected[0] = value[2]
        expected[2] = value[0]
        assert out.allclose(expected, equal_nan=True)
    else:
        assert out.allclose(value.flip(dims=[0]))

    index = torch.tensor([1], device=device)
    out = tensor.index_select(1, index)
    assert isinstance(out, HashTensor)
    assert out.size() == (3 if dtype != torch.bool else 2, 1)
    assert out.as_tensor().allclose(value[:, 1:])


@withCUDA
@withHashTensor
def test_select(device):
    key = torch.tensor([2, 1, 0], device=device)
    tensor = HashTensor(key)

    out = tensor[0]
    assert not isinstance(out, HashTensor)
    assert out.dim() == 0
    assert int(out) == 2

    out = tensor.select(0, 4)
    assert not isinstance(out, HashTensor)
    assert out.dim() == 0
    assert int(out) == -1

    with pytest.raises(IndexError, match="out of range"):
        torch.select(tensor, 1, 0)

    with pytest.raises(IndexError, match="out of range"):
        tensor.select(-2, 0)

    value = torch.randn(key.size(0), 2, device=device)
    tensor = HashTensor(key, value)

    out = tensor[0]
    assert not isinstance(out, HashTensor)
    assert out.equal(value[2])

    out = tensor.select(-1, 0)
    assert isinstance(out, HashTensor)
    assert out.as_tensor().equal(value[:, 0])


@withHashTensor
def test_tolist():
    key = torch.tensor([2, 1, 0])
    value = torch.randn(key.size(0), 2)
    assert HashTensor(key, value).tolist() == value.tolist()


@withHashTensor
def test_numpy():
    key = torch.tensor([2, 1, 0])
    value = torch.randn(key.size(0), 2)
    assert np.allclose(HashTensor(key, value).numpy(), value.numpy())


def _collate_fn(hash_tensors: List[HashTensor]) -> List[HashTensor]:
    return hash_tensors


@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('pin_memory', [False, True])
def test_data_loader(num_workers, pin_memory):
    key = torch.tensor([2, 1, 0])
    value = torch.randn(key.size(0), 2)
    tensor = HashTensor(key, value)

    loader = torch.utils.data.DataLoader(
        [tensor] * 4,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    assert len(loader) == 2
    for batch in loader:
        assert isinstance(batch, list)
        assert len(batch) == 2
        for tensor in batch:
            assert isinstance(tensor, HashTensor)
            assert tensor.dtype == value.dtype
            assert tensor.is_shared() != (num_workers == 0) or pin_memory


@withCUDA
@withHashTensor
@pytest.mark.parametrize('dtype', KEY_DTYPES[:1])
def test_getitem(dtype, device):
    if dtype != torch.bool:
        key = torch.tensor([20, 10, 0], dtype=dtype, device=device)
    else:
        key = torch.tensor([True, False], device=device)

    value = torch.randn(key.size(0), 2, 4, device=device)
    tensor = HashTensor(key, value)

    if dtype != torch.bool:
        out = tensor[10, :, None, torch.tensor([1, 2])]
    else:
        out = tensor[False, :, None, torch.tensor([1, 2])]
    assert not isinstance(out, HashTensor)
    assert out.allclose(value[1, :, None, torch.tensor([1, 2])])

    if dtype != torch.bool:
        out = tensor[..., 10, :, None, torch.tensor([1, 2])]
    else:
        out = tensor[..., False, :, None, torch.tensor([1, 2])]
    assert not isinstance(out, HashTensor)
    assert out.allclose(value[1, :, None, torch.tensor([1, 2])])

    if dtype != torch.bool:
        out = tensor[..., [10, 20], 1, None, 0:2]
    else:
        out = tensor[..., [False, True], 1, None, 0:2]
    assert not isinstance(out, HashTensor)
    assert out.allclose(value[torch.tensor([1, 0]), 1, None, 0:2])

    if dtype != torch.bool:
        out = tensor[[10, 20], 1, None, 0:2]
    else:
        out = tensor[[False, True], 1, None, 0:2]
    assert not isinstance(out, HashTensor)
    assert out.allclose(value[torch.tensor([1, 0]), 1, None, 0:2])

    out = tensor[..., None, torch.tensor([1, 2])]
    assert isinstance(out, HashTensor)
    assert out.as_tensor().allclose(value[..., None, torch.tensor([1, 2])])

    out = tensor[...]
    assert isinstance(out, HashTensor)
    assert out.size() == value.size()

    out = tensor[:2]
    assert isinstance(out, HashTensor)
    assert out.size() == (2, ) + value.size()[1:]


@onlyLinux
@withHashTensor
@withPackage('torch>=2.3')
@pytest.mark.skip(reason="Does not work currently")
def test_compile_basic():
    import torch._dynamo as dynamo

    class Model(torch.nn.Module):
        def forward(self, key: Tensor, query: Tensor) -> Tensor:
            _map = HashTensor(key)
            return _map[query]

    key = torch.randperm(10)
    query = key[:5]

    model = Model()
    expected = model(key, query)
    assert expected.equal(torch.arange(query.numel()))

    explanation = dynamo.explain(model)(key, query)
    assert explanation.graph_break_count == 0

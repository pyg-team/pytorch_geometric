import torch
from torch.nn import Linear

from torch_geometric.data import Data
from torch_geometric.profile import (
    count_parameters,
    get_cpu_memory_from_gc,
    get_data_size,
    get_gpu_memory_from_gc,
    get_gpu_memory_from_nvidia_smi,
    get_model_size,
)
from torch_geometric.profile.utils import (
    byte_to_megabyte,
    medibyte_to_megabyte,
)
from torch_geometric.testing import onlyCUDA, withPackage
from torch_geometric.typing import SparseTensor


def test_count_parameters():
    assert count_parameters(Linear(32, 128)) == 32 * 128 + 128


def test_get_model_size():
    model_size = get_model_size(Linear(32, 128, bias=False))
    assert model_size >= 32 * 128 * 4 and model_size < 32 * 128 * 4 + 2000


def test_get_data_size():
    x = torch.randn(10, 128)
    data = Data(x=x, y=x)

    data_size = get_data_size(data)
    assert data_size == 10 * 128 * 4


@withPackage('torch_sparse')
def test_get_data_size_with_sparse_tensor():
    x = torch.randn(10, 128)
    row, col = torch.randint(0, 10, (2, 100), dtype=torch.long)
    adj_t = SparseTensor(row=row, col=col, value=None, sparse_sizes=(10, 10))
    data = Data(x=x, y=x, adj_t=adj_t)

    data_size = get_data_size(data)
    assert data_size == 10 * 128 * 4 + 11 * 8 + 100 * 8


def test_get_cpu_memory_from_gc():
    old_mem = get_cpu_memory_from_gc()
    _ = torch.randn(10, 128)
    new_mem = get_cpu_memory_from_gc()
    assert new_mem - old_mem == 10 * 128 * 4


@onlyCUDA
def test_get_gpu_memory_from_gc():
    old_mem = get_gpu_memory_from_gc()
    _ = torch.randn(10, 128, device='cuda')
    new_mem = get_gpu_memory_from_gc()
    assert new_mem - old_mem == 10 * 128 * 4


@onlyCUDA
def test_get_gpu_memory_from_nvidia_smi():
    free_mem, used_mem = get_gpu_memory_from_nvidia_smi(device=0, digits=2)
    assert free_mem >= 0
    assert used_mem >= 0


def test_bytes_function():
    assert byte_to_megabyte((1024 * 1024)) == 1.00
    assert medibyte_to_megabyte(1 / 1.0485) == 1.00

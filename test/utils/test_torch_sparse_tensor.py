import torch
from torch_sparse import SparseTensor

from torch_geometric.utils import is_torch_sparse_tensor


def test_is_torch_sparse_tensor():
    x = torch.randn(5, 5)

    assert not is_torch_sparse_tensor(x)
    assert not is_torch_sparse_tensor(SparseTensor.from_dense(x))
    assert is_torch_sparse_tensor(x.to_sparse())

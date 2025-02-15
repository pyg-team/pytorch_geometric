import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.nn import DataParallel
from torch_geometric.testing import onlyCUDA


@onlyCUDA
def test_data_parallel_single_gpu():
    with pytest.warns(UserWarning, match="much slower"):
        module = DataParallel(torch.nn.Identity())
    data_list = [Data(x=torch.randn(x, 1)) for x in [2, 3, 10, 4]]
    batches = module.scatter(data_list, device_ids=[0])
    assert len(batches) == 1


@onlyCUDA
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='No multiple GPUs')
def test_data_parallel_multi_gpu():
    with pytest.warns(UserWarning, match="much slower"):
        module = DataParallel(torch.nn.Identity())
    data_list = [Data(x=torch.randn(x, 1)) for x in [2, 3, 10, 4]]
    batches = module.scatter(data_list, device_ids=[0, 1, 0, 1])
    assert len(batches) == 3

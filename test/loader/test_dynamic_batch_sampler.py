from typing import List

import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DynamicBatchSampler


def test_dataloader_with_dynamic_batches():
    data_list: List[Data] = []
    for num_nodes in range(100, 110):
        data_list.append(Data(num_nodes=num_nodes))

    torch.manual_seed(12345)
    batch_sampler = DynamicBatchSampler(data_list, 300, shuffle=True)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    num_nodes_total = 0
    for data in loader:
        assert data.num_nodes <= 300
        num_nodes_total += data.num_nodes
    assert num_nodes_total == 1045

    # Test skipping
    data_list = [Data(num_nodes=400)] + data_list
    batch_sampler = DynamicBatchSampler(data_list, 300, skip_too_big=True,
                                        num_steps=2)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    num_nodes_total = 0
    for data in loader:
        num_nodes_total += data.num_nodes
    assert num_nodes_total == 404

    # Test warning
    batch_sampler = DynamicBatchSampler(data_list, 300, skip_too_big=False,
                                        num_steps=2)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    with pytest.warns(UserWarning, match="is larger than 300 nodes"):
        num_nodes_total = 0
        for data in loader:
            num_nodes_total += data.num_nodes
        assert num_nodes_total == 601

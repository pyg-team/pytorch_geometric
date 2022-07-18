from typing import List

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DynamicBatchSampler


def test_dataloader_with_dynamic_batches():
    data_list: List[Data] = []
    for i in range(100, 110):
        data_list.append(Data(num_nodes=i))

    torch.manual_seed(12345)
    batch_sampler = DynamicBatchSampler(data_list, 300, shuffle=True)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    num_nodes_total = 0
    for data in loader:
        num_nodes_total += data.num_nodes
    assert num_nodes_total == 1045

    data_list = [Data(num_nodes=400)] + data_list
    batch_sampler = DynamicBatchSampler(data_list, 300, skip_too_big=True,
                                        num_steps=2)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    num_nodes_total = 0
    for data in loader:
        num_nodes_total += data.num_nodes
    assert num_nodes_total == 404

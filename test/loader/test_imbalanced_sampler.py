from typing import List

import torch

from torch_geometric.data import Data
from torch_geometric.loader import (
    DataLoader,
    ImbalancedSampler,
    NeighborLoader,
)


def test_dataloader_with_imbalanced_sampler():
    data_list: List[Data] = []
    for _ in range(10):
        data_list.append(Data(num_nodes=10, y=0))
    for _ in range(90):
        data_list.append(Data(num_nodes=10, y=1))

    torch.manual_seed(12345)
    sampler = ImbalancedSampler(data_list)
    loader = DataLoader(data_list, batch_size=10, sampler=sampler)

    y = torch.cat([batch.y for batch in loader])

    histogram = y.bincount()
    prob = histogram / histogram.sum()

    assert histogram.sum() == len(data_list)
    assert prob.min() > 0.4 and prob.max() < 0.6

    # Test with label tensor as input:
    torch.manual_seed(12345)
    sampler = ImbalancedSampler(torch.tensor([data.y for data in data_list]))
    loader = DataLoader(data_list, batch_size=10, sampler=sampler)

    assert torch.allclose(y, torch.cat([batch.y for batch in loader]))


def test_neighbor_loader_with_imbalanced_sampler():
    zeros = torch.zeros(10, dtype=torch.long)
    ones = torch.ones(90, dtype=torch.long)

    y = torch.cat([zeros, ones], dim=0)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(edge_index=edge_index, y=y, num_nodes=y.size(0))

    torch.manual_seed(12345)
    sampler = ImbalancedSampler(data)
    loader = NeighborLoader(data, batch_size=10, sampler=sampler,
                            num_neighbors=[-1])

    y = torch.cat([batch.y for batch in loader])

    histogram = y.bincount()
    prob = histogram / histogram.sum()

    assert histogram.sum() == data.num_nodes
    assert prob.min() > 0.4 and prob.max() < 0.6

    # Test with label tensor as input:
    torch.manual_seed(12345)
    sampler = ImbalancedSampler(data.y)
    loader = NeighborLoader(data, batch_size=10, sampler=sampler,
                            num_neighbors=[-1])

    assert torch.allclose(y, torch.cat([batch.y for batch in loader]))

import sys
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
    batch_sampler = DynamicBatchSampler(data_list, max_num_nodes=300,
                                        max_num_edges=sys.maxsize,
                                        shuffle=True)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    num_nodes_total = 0
    for data in loader:
        assert data.num_nodes <= 300
        num_nodes_total += data.num_nodes
    assert num_nodes_total == 1045

    # Test skipping
    data_list = [Data(num_nodes=400)] + data_list
    batch_sampler = DynamicBatchSampler(data_list, max_num_nodes=300,
                                        max_num_edges=sys.maxsize,
                                        skip_too_big=True, num_steps=2)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    num_nodes_total = 0
    for data in loader:
        num_nodes_total += data.num_nodes
    assert num_nodes_total == 406

    with pytest.raises(ValueError, match="length of 'DynamicBatchSampler'"):
        len(
            DynamicBatchSampler(data_list, max_num_nodes=300,
                                max_num_edges=sys.maxsize))
    assert len(
        DynamicBatchSampler(data_list, max_num_nodes=300,
                            max_num_edges=sys.maxsize, num_steps=2)) == 2


def test_dynamic_batch_dual_constraint():
    """Test that both node and edge limits are enforced simultaneously."""
    data_list: List[Data] = []
    # Each graph: 50 nodes, 200 edges
    for _ in range(6):
        edge_index = torch.zeros(2, 200, dtype=torch.long)
        data_list.append(Data(num_nodes=50, edge_index=edge_index))

    # Node limit allows 2 graphs (100 < 120), edge limit allows 2 (400 < 500)
    # So batches should be size 2
    batch_sampler = DynamicBatchSampler(data_list, max_num_nodes=120,
                                        max_num_edges=500)
    batches = [batch for batch in batch_sampler]
    for batch in batches:
        assert len(batch) <= 2

    # Now tighten edges: edge limit allows only 1 graph (200 < 300, 400 > 300)
    batch_sampler = DynamicBatchSampler(data_list, max_num_nodes=120,
                                        max_num_edges=300)
    batches = [batch for batch in batch_sampler]
    for batch in batches:
        assert len(batch) <= 1


def test_dynamic_batch_validation():
    """Test that invalid parameters raise ValueError."""
    data_list = [Data(num_nodes=10)]

    with pytest.raises(ValueError, match="max_num_nodes"):
        DynamicBatchSampler(data_list, max_num_nodes=0, max_num_edges=100)

    with pytest.raises(ValueError, match="max_num_nodes"):
        DynamicBatchSampler(data_list, max_num_nodes=-1, max_num_edges=100)

    # max_num_edges=0 is valid (fully unconnected graphs)
    DynamicBatchSampler(data_list, max_num_nodes=100, max_num_edges=0)

    with pytest.raises(ValueError, match="max_num_edges"):
        DynamicBatchSampler(data_list, max_num_nodes=100, max_num_edges=-1)


def test_dynamic_batch_skip_too_big_edges():
    """Test skip_too_big works with edge-oversized samples."""
    data_list: List[Data] = []
    # One oversized graph: 500 edges
    edge_index_big = torch.zeros(2, 500, dtype=torch.long)
    data_list.append(Data(num_nodes=5, edge_index=edge_index_big))
    # Several small graphs: 10 edges each
    for _ in range(5):
        edge_index_small = torch.zeros(2, 10, dtype=torch.long)
        data_list.append(Data(num_nodes=5, edge_index=edge_index_small))

    batch_sampler = DynamicBatchSampler(data_list, max_num_nodes=sys.maxsize,
                                        max_num_edges=100, skip_too_big=True)
    loader = DataLoader(data_list, batch_sampler=batch_sampler)

    total_edges = 0
    for data in loader:
        assert data.num_edges <= 100
        total_edges += data.num_edges
    # Should skip the 500-edge graph, only get 5 * 10 = 50
    assert total_edges == 50

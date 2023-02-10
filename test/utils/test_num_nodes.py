import torch

from torch_geometric.utils.num_nodes import (
    maybe_num_nodes,
    maybe_num_nodes_dict,
)


def test_maybe_num_nodes():
    edge_index_tensor = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
                                          [1, 2, 0, 2, 0, 1, 1]])

    assert maybe_num_nodes(edge_index_tensor, 3) == 3
    assert maybe_num_nodes(edge_index_tensor) == 3


def test_maybe_num_nodes_dict():
    edge_index_dict = {
        '1': torch.LongTensor([[0, 0, 1, 1, 2, 2, 2], [1, 2, 0, 2, 0, 1, 1]]),
        '2': torch.LongTensor([[0, 0, 1, 3], [1, 2, 0, 4]])
    }

    expected_num_nodes_dict = {'1': 3, '2': 5}

    assert maybe_num_nodes_dict(edge_index_dict) == expected_num_nodes_dict

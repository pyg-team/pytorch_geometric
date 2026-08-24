import torch

from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.utils.num_nodes import (
    maybe_num_nodes,
    maybe_num_nodes_dict,
)


def test_maybe_num_nodes():
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2], [1, 2, 0, 2, 0, 1, 1]])

    assert maybe_num_nodes(edge_index, 4) == 4
    assert maybe_num_nodes(edge_index) == 3

    adj = to_torch_coo_tensor(edge_index)
    assert maybe_num_nodes(adj, 4) == 4
    assert maybe_num_nodes(adj) == 3


def test_maybe_num_nodes_dict():
    edge_index_dict = {
        '1': torch.tensor([[0, 0, 1, 1, 2, 2, 2], [1, 2, 0, 2, 0, 1, 1]]),
        '2': torch.tensor([[0, 0, 1, 3], [1, 2, 0, 4]])
    }
    num_nodes_dict = {'2': 6}

    assert maybe_num_nodes_dict(edge_index_dict) == {'1': 3, '2': 5}
    assert maybe_num_nodes_dict(edge_index_dict, num_nodes_dict) == {
        '1': 3,
        '2': 6,
    }


def test_maybe_num_nodes_dict_empty_edge_index():
    # Test with empty edge indices (regression test for bug fix)
    edge_index_dict = {
        ('user', 'rates', 'movie'):
        torch.tensor([[], []], dtype=torch.long),
        ('user', 'follows', 'user'):
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }

    result = maybe_num_nodes_dict(edge_index_dict)
    assert result == {'user': 3, 'movie': 0}

    # Test with all empty edge indices
    edge_index_dict_all_empty = {
        ('user', 'rates', 'movie'): torch.tensor([[], []], dtype=torch.long),
        ('movie', 'in', 'genre'): torch.tensor([[], []], dtype=torch.long),
    }

    result_all_empty = maybe_num_nodes_dict(edge_index_dict_all_empty)
    assert result_all_empty == {'user': 0, 'movie': 0, 'genre': 0}

    # Test with provided num_nodes_dict and empty edges
    num_nodes_dict = {'movie': 10}
    result_with_provided = maybe_num_nodes_dict(edge_index_dict,
                                                num_nodes_dict)
    assert result_with_provided == {'user': 3, 'movie': 10}

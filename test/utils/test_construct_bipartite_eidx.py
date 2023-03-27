import torch_geometric
from torch_geometric.testing import get_random_edge_index
import pytest
import torch

def test_construct_bipartite_edge_index():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 16),
    }
    edge_index = get_random_edge_index(4, 6, num_edges=20)

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }
    combined_e_idxs = construct_bipartite_edge_index(edge_index_dict, {'author':0, 'paper':4}, {'author':0, 'paper':4})
    assert combined_e_idxs.size() == (2, 20), "combined_e_idxs.size()=" str(combined_e_idxs.size())
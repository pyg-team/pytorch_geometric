import torch

from torch_geometric.nn import FastHGTConv
from torch_geometric.testing import get_random_edge_index


def test_fast_hgt_conv():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 16),
    }
    edge_index = get_random_edge_index(4, 6, num_edges=20)

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = FastHGTConv(16, 16, metadata, heads=2)
    assert str(conv) == 'FastHGTConv(-1, 16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 16)
    assert out_dict1['paper'].size() == (6, 16)

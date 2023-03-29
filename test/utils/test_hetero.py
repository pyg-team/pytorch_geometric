import torch

from torch_geometric.testing import get_random_edge_index
from torch_geometric.utils.hetero import construct_bipartite_edge_index


def test_construct_bipartite_edge_index():
    edge_index = get_random_edge_index(4, 6, num_edges=20)

    edge_index_dict = {
        ('author', 'paper'): edge_index,
        ('paper', 'author'): edge_index.flip([0]),
    }
    edge_attr_dict = {
        ('author', 'paper'): torch.randn(edge_index.size(1), 16),
        ('paper', 'author'): torch.randn(edge_index.size(1), 16)
    }

    edge_index, edge_attr = construct_bipartite_edge_index(
        edge_index_dict,
        src_offset_dict={
            ('author', 'paper'): 0,
            ('paper', 'author'): 4
        },
        dst_offset_dict={
            'author': 0,
            'paper': 4
        },
        edge_attr_dict=edge_attr_dict,
    )

    assert edge_index.size() == (2, 40)
    assert edge_index.min() >= 0
    assert edge_index[0].max() > 4 and edge_index[1].max() > 6
    assert edge_index.max() <= 10
    assert edge_attr.size() == (40, 16)
    assert torch.equal(edge_attr[:20], edge_attr_dict['author', 'paper'])
    assert torch.equal(edge_attr[20:], edge_attr_dict['paper', 'author'])

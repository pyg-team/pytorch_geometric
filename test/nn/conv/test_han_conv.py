import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import HANConv


def test_han_conv():
    x_dict = {
        'author': torch.randn(6, 16),
        'paper': torch.randn(5, 12),
        'term': torch.randn(4, 3)
    }
    edge1 = torch.randint(0, 6, (2, 7), dtype=torch.long)
    edge2 = torch.randint(0, 5, (2, 4), dtype=torch.long)
    edge3 = torch.randint(0, 3, (2, 5), dtype=torch.long)
    edge_index_dict = {
        ('author', 'metapath0', 'author'): edge1,
        ('paper', 'metapath1', 'paper'): edge2,
        ('paper', 'metapath2', 'paper'): edge3,
    }

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12, 'term': 3}

    conv = HANConv(in_channels, 16, metadata, heads=2)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 3
    assert out_dict1['author'].size() == (6, 16)
    assert out_dict1['paper'].size() == (5, 16)
    assert out_dict1['term'] is None
    del out_dict1['term']
    del x_dict['term']

    out_dict2 = conv(x_dict, adj_t_dict)
    assert len(out_dict1) == len(out_dict2)
    for node_type in out_dict1.keys():
        assert torch.allclose(out_dict1[node_type], out_dict2[node_type],
                              atol=1e-6)

    # non zero dropout
    conv = HANConv(in_channels, 16, metadata, heads=2, dropout=0.1)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (6, 16)
    assert out_dict1['paper'].size() == (5, 16)


def test_han_conv_lazy():
    x_dict = {
        'author': torch.randn(6, 16),
        'paper': torch.randn(5, 12),
    }
    edge1 = torch.randint(0, 6, (2, 8), dtype=torch.long)
    edge2 = torch.randint(0, 5, (2, 6), dtype=torch.long)
    edge_index_dict = {
        ('author', 'to', 'author'): edge1,
        ('paper', 'to', 'paper'): edge2,
    }

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    conv = HANConv(-1, 16, metadata, heads=2)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (6, 16)
    assert out_dict1['paper'].size() == (5, 16)
    out_dict2 = conv(x_dict, adj_t_dict)

    assert len(out_dict1) == len(out_dict2)
    for node_type in out_dict1.keys():
        assert torch.allclose(out_dict1[node_type], out_dict2[node_type],
                              atol=1e-6)


def test_han_conv_empty_tensor():
    x_dict = {
        'author': torch.randn(6, 16),
        'paper': torch.empty(0, 12),
    }
    edge_index_dict = {
        ('paper', 'to', 'author'): torch.empty((2, 0), dtype=torch.long),
        ('author', 'to', 'paper'): torch.empty((2, 0), dtype=torch.long),
        ('paper', 'to', 'paper'): torch.empty((2, 0), dtype=torch.long),
    }

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12}
    conv = HANConv(in_channels, 16, metadata, heads=2)

    out_dict = conv(x_dict, edge_index_dict)
    assert len(out_dict) == 2
    assert out_dict['author'].size() == (6, 16)
    assert torch.all(out_dict['author'] == 0)
    assert out_dict['paper'].size() == (0, 16)

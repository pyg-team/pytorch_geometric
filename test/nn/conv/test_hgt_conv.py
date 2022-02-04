import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import HGTConv


def test_hgt_conv_same_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 16),
    }

    index1 = torch.randint(0, 4, (20, ), dtype=torch.long)
    index2 = torch.randint(0, 6, (20, ), dtype=torch.long)

    edge_index_dict = {
        ('author', 'writes', 'paper'): torch.stack([index1, index2]),
        ('paper', 'written_by', 'author'): torch.stack([index2, index1]),
    }

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(16, 16, metadata, heads=2)
    assert str(conv) == 'HGTConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 16)
    assert out_dict1['paper'].size() == (6, 16)
    out_dict2 = conv(x_dict, adj_t_dict)
    assert len(out_dict1) == len(out_dict2)
    for node_type in out_dict1.keys():
        assert torch.allclose(out_dict1[node_type], out_dict2[node_type],
                              atol=1e-6)

    # TODO: Test JIT functionality. We need to wait on this one until PyTorch
    # allows indexing `ParameterDict` mappings :(


def test_hgt_conv_different_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }

    index1 = torch.randint(0, 4, (20, ), dtype=torch.long)
    index2 = torch.randint(0, 6, (20, ), dtype=torch.long)

    edge_index_dict = {
        ('author', 'writes', 'paper'): torch.stack([index1, index2]),
        ('paper', 'written_by', 'author'): torch.stack([index2, index1]),
    }

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(in_channels={
        'author': 16,
        'paper': 32
    }, out_channels=32, metadata=metadata, heads=2)
    assert str(conv) == 'HGTConv(32, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 32)
    assert out_dict1['paper'].size() == (6, 32)
    out_dict2 = conv(x_dict, adj_t_dict)
    assert len(out_dict1) == len(out_dict2)
    for node_type in out_dict1.keys():
        assert torch.allclose(out_dict1[node_type], out_dict2[node_type],
                              atol=1e-6)


def test_hgt_conv_lazy():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }

    index1 = torch.randint(0, 4, (20, ), dtype=torch.long)
    index2 = torch.randint(0, 6, (20, ), dtype=torch.long)

    edge_index_dict = {
        ('author', 'writes', 'paper'): torch.stack([index1, index2]),
        ('paper', 'written_by', 'author'): torch.stack([index2, index1]),
    }

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(-1, 32, metadata, heads=2)
    assert str(conv) == 'HGTConv(32, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 32)
    assert out_dict1['paper'].size() == (6, 32)
    out_dict2 = conv(x_dict, adj_t_dict)

    assert len(out_dict1) == len(out_dict2)
    for node_type in out_dict1.keys():
        assert torch.allclose(out_dict1[node_type], out_dict2[node_type],
                              atol=1e-6)

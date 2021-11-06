import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import HANConv


def test_han_conv_same_dimensions():
    x_dict = {
        'author': torch.randn(6, 16),
    }
    metapaths = [[('author', 'paper'), ('paper', 'author')],
                 [('author', 'paper'), ('paper', 'conference'),
                  ('conference', 'paper'), ('paper', 'author')],
                 [('author', 'paper'), ('paper', 'term'),
                  ('term', 'paper'), ('paper', 'author')]]
    edge_index_dict = dict()
    for path in metapaths:
        index1 = torch.randint(0, 6, (20, ), dtype=torch.long)
        index2 = torch.randint(0, 6, (20, ), dtype=torch.long)
        src_type = path[0][0]
        rel_type = ''.join([edge_type[0][0] for edge_type in path])\
                   + path[-1][-1][0]
        dst_type = path[-1][-1]
        edge_index_dict[(src_type, rel_type, dst_type)] = torch.stack(
            [index1, index2]
        )

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HANConv(16, 16, metadata, heads=2)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 1
    assert out_dict1['author'].size() == (6, 16)
    out_dict2 = conv(x_dict, adj_t_dict)
    for out1, out2 in zip(out_dict1.values(), out_dict2.values()):
        assert torch.allclose(out1, out2, atol=1e-6)


def test_han_conv_different_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }
    metapaths = [[('author', 'paper'), ('paper', 'author')],
                 [('author', 'paper'), ('paper', 'conference'),
                  ('conference', 'paper'), ('paper', 'author')],
                 [('paper', 'author'), ('author', 'paper')],
                 [('paper', 'author'), ('author', 'conference'),
                  ('conference', 'author'), ('author', 'paper')], ]
    edge_index_dict = dict()
    for path in metapaths:
        src_type = path[0][0]
        dst_type = path[-1][-1]
        src_size = x_dict[src_type].size()[0]
        dst_size = x_dict[dst_type].size()[0]
        index1 = torch.randint(0, src_size, (20,), dtype=torch.long)
        index2 = torch.randint(0, dst_size, (20,), dtype=torch.long)
        src_type = path[0][0]
        rel_type = ''.join([edge_type[0][0] for edge_type in path])\
                   + path[-1][-1][0]
        dst_type = path[-1][-1]
        edge_index_dict[(src_type, rel_type, dst_type)] = torch.stack(
            [index1, index2]
        )

    adj_t_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict[edge_type] = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x_dict[src_type].size(0),
                          x_dict[dst_type].size(0))).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HANConv({'author': 16, 'paper': 32}, 16, metadata, heads=2)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 16)
    assert out_dict1['paper'].size() == (6, 16)
    out_dict2 = conv(x_dict, adj_t_dict)
    for out1, out2 in zip(out_dict1.values(), out_dict2.values()):
        assert torch.allclose(out1, out2, atol=1e-6)


def test_hgt_conv_lazy():
    x_dict = {
        'author': torch.randn(6, 16),
    }
    metapaths = [[('author', 'paper'), ('paper', 'author')],
                 [('author', 'paper'), ('paper', 'conference'),
                  ('conference', 'paper'), ('paper', 'author')],
                 [('author', 'paper'), ('paper', 'term'),
                  ('term', 'paper'), ('paper', 'author')]]
    edge_index_dict = dict()
    for path in metapaths:
        index1 = torch.randint(0, 6, (20,), dtype=torch.long)
        index2 = torch.randint(0, 6, (20,), dtype=torch.long)
        src_type = path[0][0]
        rel_type = ''.join([edge_type[0][0] for edge_type in path])\
                   + path[-1][-1][0]
        dst_type = path[-1][-1]
        edge_index_dict[(src_type, rel_type, dst_type)] = torch.stack(
            [index1, index2]
        )

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
    assert len(out_dict1) == 1
    assert out_dict1['author'].size() == (6, 16)
    out_dict2 = conv(x_dict, adj_t_dict)
    for out1, out2 in zip(out_dict1.values(), out_dict2.values()):
        assert torch.allclose(out1, out2, atol=1e-6)

import torch

import torch_geometric.typing
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import get_random_edge_index
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, to_torch_csc_tensor


def test_hgt_conv_same_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 16),
    }
    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_torch_csc_tensor(
            edge_index,
            size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(16, 16, metadata, heads=2)
    assert str(conv) == 'HGTConv(-1, 16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 16)
    assert out_dict1['paper'].size() == (6, 16)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert torch.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict1.keys():
            assert torch.allclose(out_dict1[key], out_dict3[key], atol=1e-6)

    # TODO: Test JIT functionality. We need to wait on this one until PyTorch
    # allows indexing `ParameterDict` mappings :(


def test_hgt_conv_different_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }
    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_torch_csc_tensor(
            edge_index,
            size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(in_channels={
        'author': 16,
        'paper': 32
    }, out_channels=32, metadata=metadata, heads=2)
    assert str(conv) == 'HGTConv(-1, 32, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 32)
    assert out_dict1['paper'].size() == (6, 32)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert torch.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for node_type in out_dict1.keys():
            assert torch.allclose(out_dict1[key], out_dict3[key], atol=1e-6)


def test_hgt_conv_lazy():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }
    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_torch_csc_tensor(
            edge_index,
            size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(-1, 32, metadata, heads=2)
    assert str(conv) == 'HGTConv(-1, 32, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 32)
    assert out_dict1['paper'].size() == (6, 32)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert torch.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if False and torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict1.keys():
            assert torch.allclose(out_dict1[key], out_dict3[key], atol=1e-6)


def test_hgt_conv_out_of_place():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)

    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    data['author', 'paper'].edge_index = edge_index
    data['paper', 'author'].edge_index = edge_index.flip([0])

    conv = HGTConv(-1, 64, data.metadata(), heads=1)

    x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
    assert x_dict['author'].size() == (4, 16)
    assert x_dict['paper'].size() == (6, 32)

    _ = conv(x_dict, edge_index_dict)

    assert x_dict['author'].size() == (4, 16)
    assert x_dict['paper'].size() == (6, 32)


def test_hgt_conv_missing_dst_node_type():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    data['author', 'paper'].edge_index = get_random_edge_index(4, 6, 20)
    data['paper', 'author'].edge_index = get_random_edge_index(6, 4, 20)
    data['university', 'author'].edge_index = get_random_edge_index(10, 4, 10)

    conv = HGTConv(-1, 64, data.metadata(), heads=1)

    out_dict = conv(data.x_dict, data.edge_index_dict)
    assert out_dict['author'].size() == (4, 64)
    assert out_dict['paper'].size() == (6, 64)
    assert 'university' not in out_dict


def test_hgt_conv_missing_input_node_type():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['author', 'writes',
         'paper'].edge_index = get_random_edge_index(4, 6, 20)

    # Some nodes from metadata are missing in data.
    # This might happen while using NeighborLoader.
    metadata = (['author', 'paper',
                 'university'], [('author', 'writes', 'paper')])
    conv = HGTConv(-1, 64, metadata, heads=1)

    out_dict = conv(data.x_dict, data.edge_index_dict)
    assert out_dict['paper'].size() == (6, 64)
    assert 'university' not in out_dict


def test_hgt_conv_missing_edge_type():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    data['author', 'writes',
         'paper'].edge_index = get_random_edge_index(4, 6, 20)

    metadata = (['author', 'paper',
                 'university'], [('author', 'writes', 'paper'),
                                 ('university', 'employs', 'author')])
    conv = HGTConv(-1, 64, metadata, heads=1)

    out_dict = conv(data.x_dict, data.edge_index_dict)
    assert out_dict['author'].size() == (4, 64)
    assert out_dict['paper'].size() == (6, 64)
    assert 'university' not in out_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    num_nodes, num_edges = 30_000, 300_000
    x_dict = {
        'paper': torch.randn(num_nodes, 64, device=args.device),
        'author': torch.randn(num_nodes, 64, device=args.device),
    }
    edge_index_dict = {
        ('paper', 'to', 'paper'):
        torch.randint(num_nodes, (2, num_edges), device=args.device),
        ('author', 'to', 'paper'):
        torch.randint(num_nodes, (2, num_edges), device=args.device),
        ('paper', 'to', 'author'):
        torch.randint(num_nodes, (2, num_edges), device=args.device),
    }

    conv = HGTConv(
        in_channels=64,
        out_channels=64,
        metadata=(list(x_dict.keys()), list(edge_index_dict.keys())),
        heads=4,
    ).to(args.device)

    benchmark(
        funcs=[conv],
        args=(x_dict, edge_index_dict),
        num_steps=10 if args.device == 'cpu' else 100,
        num_warmups=5 if args.device == 'cpu' else 50,
        backward=False,
    )

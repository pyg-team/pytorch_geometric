import copy

import torch

from torch_geometric.data import HeteroData

x_paper = torch.randn(10, 16)
x_author = torch.randn(5, 32)

idx_paper = torch.randint(x_paper.size(0), (100, ), dtype=torch.long)
idx_author = torch.randint(x_author.size(0), (100, ), dtype=torch.long)

edge_index_paper_paper = torch.stack([idx_paper[:50], idx_paper[:50]], dim=0)
edge_index_paper_author = torch.stack([idx_paper[:30], idx_author[:30]], dim=0)
edge_index_author_paper = torch.stack([idx_paper[:30], idx_author[:30]], dim=0)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def test_init_hetero_data():
    data = HeteroData()
    data['v1'].x = 1
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    assert len(data) == 2
    assert len(data.edge_types) == 3
    assert data.node_types == ['v1', 'paper', 'author']

    data = HeteroData(
        v1={'x': 1},
        paper={'x': x_paper},
        author={'x': x_author},
        paper__paper={'edge_index': edge_index_paper_paper},
        paper__author={'edge_index': edge_index_paper_author},
        author__paper={'edge_index': edge_index_author_paper},
    )
    assert len(data) == 2
    assert len(data.edge_types) == 3
    assert data.node_types == ['v1', 'paper', 'author']

    data = HeteroData({
        'v1': {
            'x': 1
        },
        'paper': {
            'x': x_paper
        },
        'author': {
            'x': x_author
        },
        ('paper', 'paper'): {
            'edge_index': edge_index_paper_paper
        },
        ('paper', 'author'): {
            'edge_index': edge_index_paper_author
        },
        ('author', 'paper'): {
            'edge_index': edge_index_author_paper
        },
    })
    assert len(data) == 2
    assert len(data.edge_types) == 3
    assert data.node_types == ['v1', 'paper', 'author']


def test_hetero_data_functions():
    data = HeteroData()
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    assert len(data) == 2
    assert sorted(data.keys) == ['edge_index', 'x']
    assert 'x' in data and 'edge_index' in data
    assert data.num_nodes == 15
    assert data.num_edges == 110

    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [
        ('paper', 'to', 'paper'),
        ('paper', 'to', 'author'),
        ('author', 'to', 'paper'),
    ]

    x_dict = data.collect('x')
    assert len(x_dict) == 2
    assert x_dict['paper'].tolist() == x_paper.tolist()
    assert x_dict['author'].tolist() == x_author.tolist()
    assert x_dict == data.x_dict

    data.y = 0
    assert data['y'] == 0 and data.y == 0
    assert len(data) == 3
    assert sorted(data.keys) == ['edge_index', 'x', 'y']

    del data['paper', 'author']
    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', 'to', 'paper'), ('author', 'to', 'paper')]

    assert len(data.to_dict()) == 5
    assert len(data.to_namedtuple()) == 5
    assert data.to_namedtuple().y == 0
    assert len(data.to_namedtuple().paper) == 1


def test_copy_hetero_data():
    data = HeteroData()
    data['paper'].x = x_paper
    data['paper', 'to', 'paper'].edge_index = edge_index_paper_paper

    out = copy.copy(data)
    assert id(data) != id(out)
    assert len(data.stores) == len(out.stores)
    for store1, store2 in zip(data.stores, out.stores):
        assert id(store1) != id(store2)
        assert id(data) == id(store1._parent())
        assert id(out) == id(store2._parent())
    assert out['paper']._key == 'paper'
    assert data['paper'].x.data_ptr() == out['paper'].x.data_ptr()
    assert out['to']._key == ('paper', 'to', 'paper')
    assert data['to'].edge_index.data_ptr() == out['to'].edge_index.data_ptr()

    out = copy.deepcopy(data)
    assert id(data) != id(out)
    assert len(data.stores) == len(out.stores)
    for store1, store2 in zip(data.stores, out.stores):
        assert id(store1) != id(store2)
    assert id(out) == id(out['paper']._parent())
    assert out['paper']._key == 'paper'
    assert data['paper'].x.data_ptr() != out['paper'].x.data_ptr()
    assert data['paper'].x.tolist() == out['paper'].x.tolist()
    assert id(out) == id(out['to']._parent())
    assert out['to']._key == ('paper', 'to', 'paper')
    assert data['to'].edge_index.data_ptr() != out['to'].edge_index.data_ptr()
    assert data['to'].edge_index.tolist() == out['to'].edge_index.tolist()


def test_to_homogeneous_and_vice_versa():
    data = HeteroData()

    data['paper'].x = torch.randn(100, 128)
    data['author'].x = torch.randn(200, 128)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 250)
    data['paper', 'paper'].edge_weight = torch.randn(250, )
    data['paper', 'paper'].edge_attr = torch.randn(250, 64)

    data['paper', 'author'].edge_index = get_edge_index(100, 200, 500)
    data['paper', 'author'].edge_weight = torch.randn(500, )
    data['paper', 'author'].edge_attr = torch.randn(500, 64)

    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)
    data['author', 'paper'].edge_weight = torch.randn(1000, )
    data['author', 'paper'].edge_attr = torch.randn(1000, 64)

    out = data.to_homogeneous()
    assert len(out) == 6
    assert out.num_nodes == 300
    assert out.num_edges == 1750
    assert out.num_node_features == 128
    assert out.num_edge_features == 64
    assert out.node_type.size() == (300, )
    assert out.node_type.min() == 0
    assert out.node_type.max() == 1
    assert out.edge_type.size() == (1750, )
    assert out.edge_type.min() == 0
    assert out.edge_type.max() == 2
    assert len(out._node_type_names) == 2
    assert len(out._edge_type_names) == 3

    out = out.to_heterogeneous()
    assert len(out) == 4
    assert torch.allclose(data['paper'].x, out['paper'].x)
    assert torch.allclose(data['author'].x, out['author'].x)

    edge_index1 = data['paper', 'paper'].edge_index
    edge_index2 = out['paper', 'paper'].edge_index
    assert edge_index1.tolist() == edge_index2.tolist()
    assert torch.allclose(
        data['paper', 'paper'].edge_weight,
        out['paper', 'paper'].edge_weight,
    )
    assert torch.allclose(
        data['paper', 'paper'].edge_attr,
        out['paper', 'paper'].edge_attr,
    )

    edge_index1 = data['paper', 'author'].edge_index
    edge_index2 = out['paper', 'author'].edge_index
    assert edge_index1.tolist() == edge_index2.tolist()
    assert torch.allclose(
        data['paper', 'author'].edge_weight,
        out['paper', 'author'].edge_weight,
    )
    assert torch.allclose(
        data['paper', 'author'].edge_attr,
        out['paper', 'author'].edge_attr,
    )

    edge_index1 = data['author', 'paper'].edge_index
    edge_index2 = out['author', 'paper'].edge_index
    assert edge_index1.tolist() == edge_index2.tolist()
    assert torch.allclose(
        data['author', 'paper'].edge_weight,
        out['author', 'paper'].edge_weight,
    )
    assert torch.allclose(
        data['author', 'paper'].edge_attr,
        out['author', 'paper'].edge_attr,
    )

    out = data.to_homogeneous()
    node_type = out.node_type
    edge_type = out.edge_type
    del out.node_type
    del out.edge_type
    del out._edge_type_names
    del out._node_type_names
    out = out.to_heterogeneous(node_type, edge_type)
    assert len(out) == 4
    assert torch.allclose(data['paper'].x, out['0'].x)
    assert torch.allclose(data['author'].x, out['1'].x)

    edge_index1 = data['paper', 'paper'].edge_index
    edge_index2 = out['0', '0'].edge_index
    assert edge_index1.tolist() == edge_index2.tolist()
    assert torch.allclose(
        data['paper', 'paper'].edge_weight,
        out['0', '0'].edge_weight,
    )
    assert torch.allclose(
        data['paper', 'paper'].edge_attr,
        out['0', '0'].edge_attr,
    )

    edge_index1 = data['paper', 'author'].edge_index
    edge_index2 = out['0', '1'].edge_index
    assert edge_index1.tolist() == edge_index2.tolist()
    assert torch.allclose(
        data['paper', 'author'].edge_weight,
        out['0', '1'].edge_weight,
    )
    assert torch.allclose(
        data['paper', 'author'].edge_attr,
        out['0', '1'].edge_attr,
    )

    edge_index1 = data['author', 'paper'].edge_index
    edge_index2 = out['1', '0'].edge_index
    assert edge_index1.tolist() == edge_index2.tolist()
    assert torch.allclose(
        data['author', 'paper'].edge_weight,
        out['1', '0'].edge_weight,
    )
    assert torch.allclose(
        data['author', 'paper'].edge_attr,
        out['1', '0'].edge_attr,
    )

    data = HeteroData()

    data['paper'].num_nodes = 100
    data['author'].num_nodes = 200

    out = data.to_homogeneous(add_node_type=False)
    assert len(out) == 1
    assert out.num_nodes == 300

    out = data.to_homogeneous().to_heterogeneous()
    assert len(out) == 1
    assert out['paper'].num_nodes == 100
    assert out['author'].num_nodes == 200

import copy

import pytest
import torch
import torch_sparse

from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage

x_paper = torch.randn(10, 16)
x_author = torch.randn(5, 32)
x_conference = torch.randn(5, 8)

idx_paper = torch.randint(x_paper.size(0), (100, ), dtype=torch.long)
idx_author = torch.randint(x_author.size(0), (100, ), dtype=torch.long)
idx_conference = torch.randint(x_conference.size(0), (100, ), dtype=torch.long)

edge_index_paper_paper = torch.stack([idx_paper[:50], idx_paper[:50]], dim=0)
edge_index_paper_author = torch.stack([idx_paper[:30], idx_author[:30]], dim=0)
edge_index_author_paper = torch.stack([idx_author[:30], idx_paper[:30]], dim=0)
edge_index_paper_conference = torch.stack(
    [idx_paper[:25], idx_conference[:25]], dim=0)

edge_attr_paper_paper = torch.randn(edge_index_paper_paper.size(1), 8)
edge_attr_author_paper = torch.randn(edge_index_author_paper.size(1), 8)


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
    data.validate(raise_on_error=True)

    assert len(data) == 2
    assert data.node_types == ['v1', 'paper', 'author']
    assert len(data.node_stores) == 3
    assert len(data.node_items()) == 3
    assert len(data.edge_types) == 3
    assert len(data.edge_stores) == 3
    assert len(data.edge_items()) == 3

    data = HeteroData(
        v1={'x': 1},
        paper={'x': x_paper},
        author={'x': x_author},
        paper__paper={'edge_index': edge_index_paper_paper},
        paper__author={'edge_index': edge_index_paper_author},
        author__paper={'edge_index': edge_index_author_paper},
    )

    assert len(data) == 2
    assert data.node_types == ['v1', 'paper', 'author']
    assert len(data.node_stores) == 3
    assert len(data.node_items()) == 3
    assert len(data.edge_types) == 3
    assert len(data.edge_stores) == 3
    assert len(data.edge_items()) == 3

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
    assert data.node_types == ['v1', 'paper', 'author']
    assert len(data.node_stores) == 3
    assert len(data.node_items()) == 3
    assert len(data.edge_types) == 3
    assert len(data.edge_stores) == 3
    assert len(data.edge_items()) == 3


def test_hetero_data_functions():
    data = HeteroData()
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    data['paper', 'paper'].edge_attr = edge_attr_paper_paper
    assert len(data) == 3
    assert sorted(data.keys) == ['edge_attr', 'edge_index', 'x']
    assert 'x' in data and 'edge_index' in data and 'edge_attr' in data
    assert data.num_nodes == 15
    assert data.num_edges == 110

    assert data.num_node_features == {'paper': 16, 'author': 32}
    assert data.num_edge_features == {
        ('paper', 'to', 'paper'): 8,
        ('paper', 'to', 'author'): 0,
        ('author', 'to', 'paper'): 0,
    }

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
    assert len(data) == 4
    assert sorted(data.keys) == ['edge_attr', 'edge_index', 'x', 'y']

    del data['paper', 'author']
    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', 'to', 'paper'), ('author', 'to', 'paper')]

    assert len(data.to_dict()) == 5
    assert len(data.to_namedtuple()) == 5
    assert data.to_namedtuple().y == 0
    assert len(data.to_namedtuple().paper) == 1


def test_hetero_data_rename():
    data = HeteroData()
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper

    data = data.rename('paper', 'article')
    assert data.node_types == ['author', 'article']
    assert data.edge_types == [
        ('article', 'to', 'article'),
        ('article', 'to', 'author'),
        ('author', 'to', 'article'),
    ]

    assert data['article'].x.tolist() == x_paper.tolist()
    edge_index = data['article', 'article'].edge_index
    assert edge_index.tolist() == edge_index_paper_paper.tolist()


def test_hetero_data_subgraph():
    data = HeteroData()
    data.num_node_types = 3
    data['paper'].x = x_paper
    data['paper'].name = 'paper'
    data['paper'].num_nodes = x_paper.size(0)
    data['author'].x = x_author
    data['author'].num_nodes = x_author.size(0)
    data['conference'].x = x_conference
    data['conference'].num_nodes = x_conference.size(0)
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'paper'].edge_attr = edge_attr_paper_paper
    data['paper', 'paper'].name = 'cites'
    data['author', 'paper'].edge_index = edge_index_author_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['paper', 'conference'].edge_index = edge_index_paper_conference

    subset = {
        'paper': torch.randperm(x_paper.size(0))[:4],
        'author': torch.randperm(x_author.size(0))[:2]
    }

    out = data.subgraph(subset)

    assert out.num_node_types == data.num_node_types
    assert out.node_types == ['paper', 'author']

    assert len(out['paper']) == 3
    assert torch.allclose(out['paper'].x, data['paper'].x[subset['paper']])
    assert out['paper'].name == 'paper'
    assert out['paper'].num_nodes == 4
    assert len(out['author']) == 2
    assert torch.allclose(out['author'].x, data['author'].x[subset['author']])
    assert out['author'].num_nodes == 2

    assert out.edge_types == [
        ('paper', 'to', 'paper'),
        ('author', 'to', 'paper'),
        ('paper', 'to', 'author'),
    ]

    assert len(out['paper', 'paper']) == 3
    assert out['paper', 'paper'].edge_index is not None
    assert out['paper', 'paper'].edge_attr is not None
    assert out['paper', 'paper'].name == 'cites'
    assert len(out['paper', 'author']) == 1
    assert out['paper', 'author'].edge_index is not None
    assert len(out['author', 'paper']) == 1
    assert out['author', 'paper'].edge_index is not None

    out = data.node_type_subgraph(['paper', 'author'])
    assert out.node_types == ['paper', 'author']
    assert out.edge_types == [('paper', 'to', 'paper'),
                              ('author', 'to', 'paper'),
                              ('paper', 'to', 'author')]

    out = data.edge_type_subgraph([('paper', 'author')])
    assert out.node_types == ['paper', 'author']
    assert out.edge_types == [('paper', 'to', 'author')]


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
    data['paper'].y = torch.randint(0, 10, (100, ))
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
    assert len(out) == 7
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
    assert out.y.size() == (300, )
    assert torch.allclose(out.y[:100], data['paper'].y)
    assert torch.all(out.y[100:] == -1)
    assert 'y' not in data['author']

    out = out.to_heterogeneous()
    assert len(out) == 5
    assert torch.allclose(data['paper'].x, out['paper'].x)
    assert torch.allclose(data['author'].x, out['author'].x)
    assert torch.allclose(data['paper'].y, out['paper'].y)

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
    assert len(out) == 5
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


def test_hetero_data_to_canonical():
    data = HeteroData()
    assert isinstance(data['user', 'product'], EdgeStorage)
    assert len(data.edge_types) == 1
    assert isinstance(data['user', 'to', 'product'], EdgeStorage)
    assert len(data.edge_types) == 1

    data = HeteroData()
    assert isinstance(data['user', 'buys', 'product'], EdgeStorage)
    assert isinstance(data['user', 'clicks', 'product'], EdgeStorage)
    assert len(data.edge_types) == 2

    with pytest.raises(TypeError, match="missing 1 required"):
        data['user', 'product']


# Feature Store ###############################################################


def test_basic_feature_store():
    data = HeteroData()
    x = torch.randn(20, 20)

    # Put tensor:
    assert data.put_tensor(copy.deepcopy(x), group_name='paper', attr_name='x',
                           index=None)
    assert torch.equal(data['paper'].x, x)

    # Put (modify) tensor slice:
    x[15:] = 0
    data.put_tensor(0, group_name='paper', attr_name='x',
                    index=slice(15, None, None))

    # Get tensor:
    out = data.get_tensor(group_name='paper', attr_name='x', index=None)
    assert torch.equal(x, out)

    # Get tensor size:
    assert data.get_tensor_size(group_name='paper', attr_name='x') == (20, 20)

    # Get tensor attrs:
    data['paper'].num_nodes = 20  # don't include, not a tensor attr
    data['paper'].bad_attr = torch.randn(10, 20)  # don't include, bad cat_dim

    tensor_attrs = data.get_all_tensor_attrs()
    assert len(tensor_attrs) == 1
    assert tensor_attrs[0].group_name == 'paper'
    assert tensor_attrs[0].attr_name == 'x'

    # Remove tensor:
    assert 'x' in data['paper'].__dict__['_mapping']
    data.remove_tensor(group_name='paper', attr_name='x', index=None)
    assert 'x' not in data['paper'].__dict__['_mapping']


# Graph Store #################################################################


def test_basic_graph_store():
    data = HeteroData()

    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1],
                                    sparse_sizes=(3, 3))

    def assert_equal_tensor_tuple(expected, actual):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            assert torch.equal(expected[i], actual[i])

    # We put all three tensor types: COO, CSR, and CSC, and we get them back
    # to confirm that `GraphStore` works as intended.
    coo = adj.coo()[:-1]
    csr = adj.csr()[:-1]
    csc = adj.csc()[-2::-1]  # (row, colptr)

    # Put:
    data.put_edge_index(coo, layout='coo', edge_type=('a', 'to', 'b'),
                        size=(3, 3))
    data.put_edge_index(csr, layout='csr', edge_type=('a', 'to', 'c'),
                        size=(3, 3))
    data.put_edge_index(csc, layout='csc', edge_type=('b', 'to', 'c'),
                        size=(3, 3))

    # Get:
    assert_equal_tensor_tuple(
        coo, data.get_edge_index(layout='coo', edge_type=('a', 'to', 'b')))
    assert_equal_tensor_tuple(
        csr, data.get_edge_index(layout='csr', edge_type=('a', 'to', 'c')))
    assert_equal_tensor_tuple(
        csc, data.get_edge_index(layout='csc', edge_type=('b', 'to', 'c')))

    # Get attrs:
    edge_attrs = data.get_all_edge_attrs()
    assert len(edge_attrs) == 3

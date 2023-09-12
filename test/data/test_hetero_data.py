import copy

import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.testing import get_random_edge_index, withPackage

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


def test_init_hetero_data():
    data = HeteroData()
    data['v1'].x = 1
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    with pytest.warns(UserWarning, match="{'v1'} are isolated"):
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


def test_hetero_data_to_from_dict():
    data = HeteroData()
    data.global_id = '1'
    data['v1'].x = torch.randn(5, 16)
    data['v2'].y = torch.randn(4, 16)
    data['v1', 'v2'].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

    out = HeteroData.from_dict(data.to_dict())
    assert out.global_id == data.global_id
    assert torch.equal(out['v1'].x, data['v1'].x)
    assert torch.equal(out['v2'].y, data['v2'].y)
    assert torch.equal(out['v1', 'v2'].edge_index, data['v1', 'v2'].edge_index)


def test_hetero_data_functions():
    data = HeteroData()
    with pytest.raises(KeyError, match="did not find any occurrences of it"):
        data.collect('x')
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    data['paper', 'paper'].edge_attr = edge_attr_paper_paper
    assert len(data) == 3
    assert sorted(data.keys()) == ['edge_attr', 'edge_index', 'x']
    assert 'x' in data and 'edge_index' in data and 'edge_attr' in data
    assert data.num_nodes == 15
    assert data.num_edges == 110

    assert data.node_attrs() == ['x']
    assert sorted(data.edge_attrs()) == ['edge_attr', 'edge_index']

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
    assert sorted(data.keys()) == ['edge_attr', 'edge_index', 'x', 'y']

    del data['paper', 'author']
    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', 'to', 'paper'), ('author', 'to', 'paper')]

    assert len(data.to_dict()) == 5
    assert len(data.to_namedtuple()) == 5
    assert data.to_namedtuple().y == 0
    assert len(data.to_namedtuple().paper) == 1


def test_hetero_data_set_value_dict():
    data = HeteroData()
    data.set_value_dict('x', {
        'paper': torch.randn(4, 16),
        'author': torch.randn(8, 32),
    })
    assert data.node_types == ['paper', 'author']
    assert data.edge_types == []
    assert data['paper'].x.size() == (4, 16)
    assert data['author'].x.size() == (8, 32)


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


def test_dangling_types():
    data = HeteroData()
    data['src', 'to', 'dst'].edge_index = torch.randint(0, 10, (2, 20))
    with pytest.raises(ValueError, match="do not exist as node types"):
        data.validate()

    data = HeteroData()
    data['node'].num_nodes = 10
    with pytest.warns(UserWarning, match="{'node'} are isolated"):
        data.validate()


def test_hetero_data_subgraph():
    data = HeteroData()
    data.num_node_types = 3
    data['paper'].x = x_paper
    data['paper'].name = 'paper'
    data['paper'].num_nodes = x_paper.size(0)
    data['author'].x = x_author
    data['author'].num_nodes = x_author.size(0)
    data['conf'].x = x_conference
    data['conf'].num_nodes = x_conference.size(0)
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'paper'].edge_attr = edge_attr_paper_paper
    data['paper', 'paper'].name = 'cites'
    data['author', 'paper'].edge_index = edge_index_author_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['paper', 'conf'].edge_index = edge_index_paper_conference

    subset = {
        'paper': torch.randperm(x_paper.size(0))[:4],
        'author': torch.randperm(x_author.size(0))[:2],
        'conf': torch.randperm(x_conference.size(0))[:2],
    }

    out = data.subgraph(subset)
    out.validate(raise_on_error=True)

    assert out.num_node_types == data.num_node_types
    assert out.node_types == ['paper', 'author', 'conf']

    for key in out.node_types:
        assert len(out[key]) == len(data[key])
        assert torch.allclose(out[key].x, data[key].x[subset[key]])
        assert out[key].num_nodes == subset[key].size(0)
        if key == 'paper':
            assert out['paper'].name == 'paper'

    # Construct correct edge index manually:
    node_mask = {}  # for each node type a mask of nodes in the subgraph
    node_map = {}  # for each node type a map from old node id to new node id
    for key in out.node_types:
        node_mask[key] = torch.zeros((data[key].num_nodes, ), dtype=torch.bool)
        node_map[key] = torch.zeros((data[key].num_nodes, ), dtype=torch.long)
        node_mask[key][subset[key]] = True
        node_map[key][subset[key]] = torch.arange(subset[key].size(0))

    edge_mask = {}  # for each edge type a mask of edges in the subgraph
    subgraph_edge_index = {
    }  # for each edge type the edge index of the subgraph
    for key in out.edge_types:
        edge_mask[key] = (node_mask[key[0]][data[key].edge_index[0]]
                          & node_mask[key[-1]][data[key].edge_index[1]])
        subgraph_edge_index[key] = data[key].edge_index[:, edge_mask[key]]
        subgraph_edge_index[key][0] = node_map[key[0]][subgraph_edge_index[key]
                                                       [0]]
        subgraph_edge_index[key][1] = node_map[key[-1]][
            subgraph_edge_index[key][1]]

    assert out.edge_types == [
        ('paper', 'to', 'paper'),
        ('author', 'to', 'paper'),
        ('paper', 'to', 'author'),
        ('paper', 'to', 'conf'),
    ]

    for key in out.edge_types:
        assert len(out[key]) == len(data[key])
        assert torch.equal(out[key].edge_index, subgraph_edge_index[key])
        if key == ('paper', 'to', 'paper'):
            assert torch.allclose(out[key].edge_attr,
                                  data[key].edge_attr[edge_mask[key]])
            assert out[key].name == 'cites'

    # Test for bool and long in `subset_dict`.
    author_mask = torch.zeros((x_author.size(0), ), dtype=torch.bool)
    author_mask[subset['author']] = True
    subset_mixed = {
        'paper': subset['paper'],
        'author': author_mask,
    }
    out = data.subgraph(subset_mixed)
    out.validate(raise_on_error=True)

    assert out.num_node_types == data.num_node_types
    assert out.node_types == ['paper', 'author', 'conf']
    assert out['paper'].num_nodes == subset['paper'].size(0)
    assert out['author'].num_nodes == subset['author'].size(0)
    assert out['conf'].num_nodes == data['conf'].num_nodes
    assert out.edge_types == [
        ('paper', 'to', 'paper'),
        ('author', 'to', 'paper'),
        ('paper', 'to', 'author'),
        ('paper', 'to', 'conf'),
    ]

    out = data.node_type_subgraph(['paper', 'author'])
    assert out.node_types == ['paper', 'author']
    assert out.edge_types == [('paper', 'to', 'paper'),
                              ('author', 'to', 'paper'),
                              ('paper', 'to', 'author')]

    out = data.edge_type_subgraph([('paper', 'author')])
    assert out.node_types == ['paper', 'author']
    assert out.edge_types == [('paper', 'to', 'author')]

    subset = {
        ('paper', 'to', 'paper'): torch.arange(4),
    }

    out = data.edge_subgraph(subset)
    assert out.node_types == data.node_types
    assert out.edge_types == data.edge_types
    assert data['paper'] == out['paper']
    assert data['author'] == out['author']
    assert data['paper', 'author'] == out['paper', 'author']
    assert data['author', 'paper'] == out['author', 'paper']

    assert out['paper', 'paper'].num_edges == 4
    assert out['paper', 'paper'].edge_index.size() == (2, 4)
    assert out['paper', 'paper'].edge_attr.size() == (4, 8)


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

    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 250)
    data['paper', 'paper'].edge_weight = torch.randn(250, )
    data['paper', 'paper'].edge_attr = torch.randn(250, 64)

    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 500)
    data['paper', 'author'].edge_weight = torch.randn(500, )
    data['paper', 'author'].edge_attr = torch.randn(500, 64)

    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 1000)
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


def test_to_homogeneous_padding():
    data = HeteroData()
    data['paper'].x = torch.randn(100, 128)
    data['author'].x = torch.randn(50, 64)

    out = data.to_homogeneous()
    assert len(out) == 2
    assert out.node_type.size() == (150, )
    assert out.node_type[:100].abs().sum() == 0
    assert out.node_type[100:].sub(1).abs().sum() == 0
    assert out.x.size() == (150, 128)
    assert torch.equal(out.x[:100], data['paper'].x)
    assert torch.equal(out.x[100:, :64], data['author'].x)
    assert out.x[100:, 64:].abs().sum() == 0


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


def test_hetero_data_invalid_names():
    data = HeteroData()
    with pytest.warns(UserWarning, match="single underscores"):
        data['my test', 'a__b', 'my test'].edge_attr = torch.randn(10, 16)
    assert data.edge_types == [('my test', 'a__b', 'my test')]


def test_hetero_data_update():
    data = HeteroData()
    data['paper'].x = torch.arange(0, 5)
    data['paper'].y = torch.arange(5, 10)
    data['author'].x = torch.arange(10, 15)

    other = HeteroData()
    other['paper'].x = torch.arange(15, 20)
    other['author'].y = torch.arange(20, 25)
    other['paper', 'paper'].edge_index = torch.randint(5, (2, 20))

    data.update(other)
    assert len(data) == 3
    assert torch.equal(data['paper'].x, torch.arange(15, 20))
    assert torch.equal(data['paper'].y, torch.arange(5, 10))
    assert torch.equal(data['author'].x, torch.arange(10, 15))
    assert torch.equal(data['author'].y, torch.arange(20, 25))
    assert torch.equal(data['paper', 'paper'].edge_index,
                       other['paper', 'paper'].edge_index)


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


@withPackage('torch_sparse')
def test_basic_graph_store():
    data = HeteroData()

    def assert_equal_tensor_tuple(expected, actual):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            assert torch.equal(expected[i], actual[i])

    # We put all three tensor types: COO, CSR, and CSC, and we get them back
    # to confirm that `GraphStore` works as intended.
    coo = (torch.tensor([0, 1]), torch.tensor([1, 2]))
    csr = (torch.tensor([0, 1, 2, 2]), torch.tensor([1, 2]))
    csc = (torch.tensor([0, 1]), torch.tensor([0, 0, 1, 2]))

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


def test_data_generate_ids():
    data = HeteroData()

    data['paper'].x = torch.randn(100, 128)
    data['author'].x = torch.randn(200, 128)

    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 300)
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 400)
    assert len(data) == 2

    data.generate_ids()
    assert len(data) == 4
    assert data['paper'].n_id.tolist() == list(range(100))
    assert data['author'].n_id.tolist() == list(range(200))
    assert data['paper', 'author'].e_id.tolist() == list(range(300))
    assert data['author', 'paper'].e_id.tolist() == list(range(400))

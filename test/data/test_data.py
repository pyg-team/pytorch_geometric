import copy

import pytest
import torch
import torch.multiprocessing as mp

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.storage import AttrType
from torch_geometric.testing import withPackage


def test_data():
    torch_geometric.set_debug(True)

    x = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float).t()
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index).to(torch.device('cpu'))
    data.validate(raise_on_error=True)

    N = data.num_nodes
    assert N == 3

    assert data.node_attrs() == ['x']
    assert data.edge_attrs() == ['edge_index']

    assert data.x.tolist() == x.tolist()
    assert data['x'].tolist() == x.tolist()
    assert data.get('x').tolist() == x.tolist()
    assert data.get('y', 2) == 2
    assert data.get('y', None) is None
    assert data.num_edge_types == 1
    assert data.num_node_types == 1
    assert next(data('x')) == ('x', x)

    assert sorted(data.keys()) == ['edge_index', 'x']
    assert len(data) == 2
    assert 'x' in data and 'edge_index' in data and 'pos' not in data

    data.apply_(lambda x: x.mul_(2), 'x')
    assert torch.allclose(data.x, x)

    data.requires_grad_('x')
    assert data.x.requires_grad is True

    D = data.to_dict()
    assert len(D) == 2
    assert 'x' in D and 'edge_index' in D

    D = data.to_namedtuple()
    assert len(D) == 2
    assert D.x is not None and D.edge_index is not None

    assert data.__cat_dim__('x', data.x) == 0
    assert data.__cat_dim__('edge_index', data.edge_index) == -1
    assert data.__inc__('x', data.x) == 0
    assert data.__inc__('edge_index', data.edge_index) == data.num_nodes

    assert not data.x.is_contiguous()
    data.contiguous()
    assert data.x.is_contiguous()

    assert not data.is_coalesced()
    data = data.coalesce()
    assert data.is_coalesced()

    clone = data.clone()
    assert clone != data
    assert len(clone) == len(data)
    assert clone.x.data_ptr() != data.x.data_ptr()
    assert clone.x.tolist() == data.x.tolist()
    assert clone.edge_index.data_ptr() != data.edge_index.data_ptr()
    assert clone.edge_index.tolist() == data.edge_index.tolist()

    # Test `data.to_heterogeneous()`:
    out = data.to_heterogeneous()
    assert torch.allclose(data.x, out['0'].x)
    assert torch.allclose(data.edge_index, out['0', '0'].edge_index)

    data.edge_type = torch.tensor([0, 0, 1, 0])
    out = data.to_heterogeneous()
    assert torch.allclose(data.x, out['0'].x)
    assert [store.num_edges for store in out.edge_stores] == [3, 1]
    data.edge_type = None

    data['x'] = x + 1
    assert data.x.tolist() == (x + 1).tolist()

    assert str(data) == 'Data(x=[3, 2], edge_index=[2, 4])'

    dictionary = {'x': data.x, 'edge_index': data.edge_index}
    data = Data.from_dict(dictionary)
    assert sorted(data.keys()) == ['edge_index', 'x']

    assert not data.has_isolated_nodes()
    assert not data.has_self_loops()
    assert data.is_undirected()
    assert not data.is_directed()

    assert data.num_nodes == 3
    assert data.num_edges == 4
    with pytest.warns(UserWarning, match='deprecated'):
        assert data.num_faces is None
    assert data.num_node_features == 2
    assert data.num_features == 2

    data.edge_attr = torch.randn(data.num_edges, 2)
    assert data.num_edge_features == 2
    data.edge_attr = None

    data.x = None
    with pytest.warns(UserWarning, match='Unable to accurately infer'):
        assert data.num_nodes == 3

    data.edge_index = None
    with pytest.warns(UserWarning, match='Unable to accurately infer'):
        assert data.num_nodes is None
    assert data.num_edges == 0

    data.num_nodes = 4
    assert data.num_nodes == 4

    data = Data(x=x, attribute=x)
    assert len(data) == 2
    assert data.x.tolist() == x.tolist()
    assert data.attribute.tolist() == x.tolist()

    face = torch.tensor([[0, 1], [1, 2], [2, 3]])
    data = Data(num_nodes=4, face=face)
    with pytest.warns(UserWarning, match='deprecated'):
        assert data.num_faces == 2
    assert data.num_nodes == 4

    data = Data(title='test')
    assert str(data) == "Data(title='test')"
    assert data.num_node_features == 0
    assert data.num_edge_features == 0

    key = value = 'test_value'
    data[key] = value
    assert data[key] == value
    del data[value]
    del data[value]  # Deleting unset attributes should work as well.

    assert data.get(key) is None
    assert data.get('title') == 'test'

    torch_geometric.set_debug(False)


def test_data_attr_cache():
    x = torch.randn(3, 16)
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    edge_attr = torch.randn(5, 4)
    y = torch.tensor([0])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    assert data.is_node_attr('x')
    assert 'x' in data._store._cached_attr[AttrType.NODE]
    assert 'x' not in data._store._cached_attr[AttrType.EDGE]
    assert 'x' not in data._store._cached_attr[AttrType.OTHER]

    assert not data.is_node_attr('edge_index')
    assert 'edge_index' not in data._store._cached_attr[AttrType.NODE]
    assert 'edge_index' in data._store._cached_attr[AttrType.EDGE]
    assert 'edge_index' not in data._store._cached_attr[AttrType.OTHER]

    assert data.is_edge_attr('edge_attr')
    assert 'edge_attr' not in data._store._cached_attr[AttrType.NODE]
    assert 'edge_attr' in data._store._cached_attr[AttrType.EDGE]
    assert 'edge_attr' not in data._store._cached_attr[AttrType.OTHER]

    assert not data.is_edge_attr('y')
    assert 'y' not in data._store._cached_attr[AttrType.NODE]
    assert 'y' not in data._store._cached_attr[AttrType.EDGE]
    assert 'y' in data._store._cached_attr[AttrType.OTHER]


def test_to_heterogeneous_empty_edge_index():
    data = Data(
        x=torch.randn(5, 10),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    hetero_data = data.to_heterogeneous()
    assert hetero_data.node_types == ['0']
    assert hetero_data.edge_types == []
    assert len(hetero_data) == 1
    assert torch.equal(hetero_data['0'].x, data.x)

    hetero_data = data.to_heterogeneous(
        node_type_names=['0'],
        edge_type_names=[('0', 'to', '0')],
    )
    assert hetero_data.node_types == ['0']
    assert hetero_data.edge_types == [('0', 'to', '0')]
    assert len(hetero_data) == 2
    assert torch.equal(hetero_data['0'].x, data.x)
    assert torch.equal(hetero_data['0', 'to', '0'].edge_index, data.edge_index)


def test_data_subgraph():
    x = torch.arange(5)
    y = torch.tensor([0.])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]])
    edge_weight = torch.arange(edge_index.size(1))

    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight,
                num_nodes=5)

    out = data.subgraph(torch.tensor([1, 2, 3]))
    assert len(out) == 5
    assert torch.equal(out.x, torch.arange(1, 4))
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert torch.equal(out.edge_weight, edge_weight[torch.arange(2, 6)])
    assert out.num_nodes == 3

    # Test unordered selection:
    out = data.subgraph(torch.tensor([3, 1, 2]))
    assert len(out) == 5
    assert torch.equal(out.x, torch.tensor([3, 1, 2]))
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[1, 2, 2, 0], [2, 1, 0, 2]]
    assert torch.equal(out.edge_weight, edge_weight[torch.arange(2, 6)])
    assert out.num_nodes == 3

    out = data.subgraph(torch.tensor([False, False, False, True, True]))
    assert len(out) == 5
    assert torch.equal(out.x, torch.arange(3, 5))
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[0, 1], [1, 0]]
    assert torch.equal(out.edge_weight, edge_weight[torch.arange(6, 8)])
    assert out.num_nodes == 2

    out = data.edge_subgraph(torch.tensor([1, 2, 3]))
    assert len(out) == 5
    assert out.num_nodes == data.num_nodes
    assert torch.equal(out.x, data.x)
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[1, 1, 2], [0, 2, 1]]
    assert torch.equal(out.edge_weight, edge_weight[torch.tensor([1, 2, 3])])

    out = data.edge_subgraph(
        torch.tensor([False, True, True, True, False, False, False, False]))
    assert len(out) == 5
    assert out.num_nodes == data.num_nodes
    assert torch.equal(out.x, data.x)
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[1, 1, 2], [0, 2, 1]]
    assert torch.equal(out.edge_weight, edge_weight[torch.tensor([1, 2, 3])])


def test_data_subgraph_with_list_field():
    x = torch.arange(5)
    y = list(range(5))
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]])
    data = Data(x=x, y=y, edge_index=edge_index)

    out = data.subgraph(torch.tensor([1, 2, 3]))
    assert len(out) == 3
    assert out.x.tolist() == out.y == [1, 2, 3]

    out = data.subgraph(torch.tensor([False, True, True, True, False]))
    assert len(out) == 3
    assert out.x.tolist() == out.y == [1, 2, 3]


def test_copy_data():
    data = Data(x=torch.randn(20, 5))

    out = copy.copy(data)
    assert id(data) != id(out)
    assert id(data._store) != id(out._store)
    assert len(data.stores) == len(out.stores)
    for store1, store2 in zip(data.stores, out.stores):
        assert id(store1) != id(store2)
        assert id(data) == id(store1._parent())
        assert id(out) == id(store2._parent())
    assert data.x.data_ptr() == out.x.data_ptr()

    out = copy.deepcopy(data)
    assert id(data) != id(out)
    assert id(data._store) != id(out._store)
    assert len(data.stores) == len(out.stores)
    for store1, store2 in zip(data.stores, out.stores):
        assert id(store1) != id(store2)
        assert id(data) == id(store1._parent())
        assert id(out) == id(store2._parent())
    assert data.x.data_ptr() != out.x.data_ptr()
    assert data.x.tolist() == out.x.tolist()


def test_data_sort():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 2, 1, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(6, 8)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    assert not data.is_sorted(sort_by_row=True)
    assert not data.is_sorted(sort_by_row=False)

    out = data.sort(sort_by_row=True)
    assert out.is_sorted(sort_by_row=True)
    assert not out.is_sorted(sort_by_row=False)
    assert torch.equal(out.x, data.x)
    assert out.edge_index.tolist() == [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]
    assert torch.equal(
        out.edge_attr,
        data.edge_attr[torch.tensor([0, 1, 2, 4, 3, 5])],
    )

    out = data.sort(sort_by_row=False)
    assert not out.is_sorted(sort_by_row=True)
    assert out.is_sorted(sort_by_row=False)
    assert torch.equal(out.x, data.x)
    assert out.edge_index.tolist() == [[1, 2, 3, 0, 0, 0], [0, 0, 0, 1, 2, 3]]
    assert torch.equal(
        out.edge_attr,
        data.edge_attr[torch.tensor([4, 3, 5, 0, 1, 2])],
    )


def test_debug_data():
    torch_geometric.set_debug(True)

    Data()
    Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
    Data(face=torch.zeros((3, 0), dtype=torch.long), num_nodes=10)
    Data(edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2))
    Data(x=torch.torch.randn(5, 3), num_nodes=5)
    Data(pos=torch.torch.randn(5, 3), num_nodes=5)
    Data(norm=torch.torch.randn(5, 3), num_nodes=5)

    torch_geometric.set_debug(False)


def run(rank, data_list):
    for data in data_list:
        assert data.x.is_shared()
        data.x.add_(1)


def test_data_share_memory():
    data_list = [Data(x=torch.zeros(8)) for _ in range(10)]

    for data in data_list:
        assert not data.x.is_shared()
        assert torch.all(data.x == 0.0)

    mp.spawn(run, args=(data_list, ), nprocs=4, join=True)

    for data in data_list:
        assert data.x.is_shared()
        assert torch.all(data.x > 0.0)


def test_data_setter_properties():
    class MyData(Data):
        def __init__(self):
            super().__init__()
            self.my_attr1 = 1
            self.my_attr2 = 2

        @property
        def my_attr1(self):
            return self._my_attr1

        @my_attr1.setter
        def my_attr1(self, value):
            self._my_attr1 = value

    data = MyData()
    assert data.my_attr2 == 2

    assert 'my_attr1' not in data._store
    assert data.my_attr1 == 1

    data.my_attr1 = 2
    assert 'my_attr1' not in data._store
    assert data.my_attr1 == 2


def test_data_update():
    data = Data(x=torch.arange(0, 5), y=torch.arange(5, 10))
    other = Data(z=torch.arange(10, 15), x=torch.arange(15, 20))
    data.update(other)

    assert len(data) == 3
    assert torch.equal(data.x, torch.arange(15, 20))
    assert torch.equal(data.y, torch.arange(5, 10))
    assert torch.equal(data.z, torch.arange(10, 15))


# Feature Store ###############################################################


def test_basic_feature_store():
    data = Data()
    x = torch.randn(20, 20)
    data.not_a_tensor_attr = 10  # don't include, not a tensor attr
    data.bad_attr = torch.randn(10, 20)  # don't include, bad cat_dim

    # Put tensor:
    assert data.put_tensor(copy.deepcopy(x), attr_name='x', index=None)
    assert torch.equal(data.x, x)

    # Put (modify) tensor slice:
    x[15:] = 0
    data.put_tensor(0, attr_name='x', index=slice(15, None, None))

    # Get tensor:
    out = data.get_tensor(attr_name='x', index=None)
    assert torch.equal(x, out)

    # Get tensor size:
    assert data.get_tensor_size(attr_name='x') == (20, 20)

    # Get tensor attrs:
    tensor_attrs = data.get_all_tensor_attrs()
    assert len(tensor_attrs) == 1
    assert tensor_attrs[0].attr_name == 'x'

    # Remove tensor:
    assert 'x' in data.__dict__['_store']
    data.remove_tensor(attr_name='x', index=None)
    assert 'x' not in data.__dict__['_store']


# Graph Store #################################################################


@withPackage('torch_sparse')
def test_basic_graph_store():
    r"""Test the core graph store API."""
    data = Data()

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
    data.put_edge_index(coo, layout='coo', size=(3, 3))
    data.put_edge_index(csr, layout='csr')
    data.put_edge_index(csc, layout='csc')

    # Get:
    assert_equal_tensor_tuple(coo, data.get_edge_index('coo'))
    assert_equal_tensor_tuple(csr, data.get_edge_index('csr'))
    assert_equal_tensor_tuple(csc, data.get_edge_index('csc'))

    # Get attrs:
    edge_attrs = data.get_all_edge_attrs()
    assert len(edge_attrs) == 3

    # Remove:
    coo, csr, csc = edge_attrs
    data.remove_edge_index(coo)
    data.remove_edge_index(csr)
    data.remove_edge_index(csc)
    assert len(data.get_all_edge_attrs()) == 0


def test_data_generate_ids():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])

    data = Data(x=x, edge_index=edge_index)
    assert len(data) == 2

    data.generate_ids()
    assert len(data) == 4
    assert data.n_id.tolist() == [0, 1, 2]
    assert data.e_id.tolist() == [0, 1, 2, 3, 4]


@withPackage('torch_frame')
def test_data_with_tensor_frame(get_tensor_frame):
    tf = get_tensor_frame(num_rows=10)
    data = Data(tf=tf, edge_index=torch.randint(0, 10, size=(2, 20)))

    # Test basic attributes:
    assert data.is_node_attr('tf')
    assert data.num_nodes == tf.num_rows
    assert data.num_edges == 20
    assert data.num_node_features == tf.num_cols

    # Test subgraph:
    index = torch.tensor([1, 2, 3])
    sub_data = data.subgraph(index)
    assert sub_data.num_nodes == 3
    for key, value in sub_data.tf.feat_dict.items():
        assert torch.allclose(value, tf.feat_dict[key][index])

    mask = torch.tensor(
        [False, True, True, True, False, False, False, False, False, False])
    data_sub = data.subgraph(mask)
    assert data_sub.num_nodes == 3
    for key, value in sub_data.tf.feat_dict.items():
        assert torch.allclose(value, tf.feat_dict[key][mask])

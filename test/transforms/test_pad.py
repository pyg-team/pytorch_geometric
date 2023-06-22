import numbers
from copy import deepcopy
from typing import Dict, Generator, List, Optional, Tuple, Union

import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.transforms import Pad
from torch_geometric.transforms.pad import (
    AttrNamePadding,
    EdgeTypePadding,
    NodeTypePadding,
    Padding,
    UniformPadding,
)
from torch_geometric.typing import EdgeType, NodeType


def fake_data() -> Data:
    return FakeDataset(avg_num_nodes=10, avg_degree=5, edge_dim=2)[0]


def fake_hetero_data(node_types=2, edge_types=5) -> HeteroData:
    return FakeHeteroDataset(num_node_types=node_types,
                             num_edge_types=edge_types, avg_num_nodes=10,
                             edge_dim=2)[0]


def _generate_homodata_node_attrs(data: Data) -> Generator[str, None, None]:
    for attr in data.keys():
        if data.is_node_attr(attr):
            yield attr


def _generate_homodata_edge_attrs(data: Data) -> Generator[str, None, None]:
    for attr in data.keys():
        if data.is_edge_attr(attr):
            yield attr


def _generate_heterodata_nodes(
    data: HeteroData
) -> Generator[Tuple[NodeType, str, torch.Tensor], None, None]:
    for node_type, store in data.node_items():
        for attr in store.keys():
            yield node_type, attr


def _generate_heterodata_edges(
    data: HeteroData
) -> Generator[Tuple[EdgeType, str, torch.Tensor], None, None]:
    for edge_type, store in data.edge_items():
        for attr in store.keys():
            yield edge_type, attr


def _check_homo_data_nodes(
    original: Data,
    padded: Data,
    max_num_nodes: Union[int, Dict[NodeType, int]],
    node_pad_value: Optional[Padding] = None,
    is_mask_available: bool = False,
    exclude_keys: Optional[List[str]] = None,
):
    assert padded.num_nodes == max_num_nodes

    compare_pad_start_idx = original.num_nodes

    if is_mask_available:
        assert padded.pad_node_mask.numel() == padded.num_nodes
        assert torch.all(padded.pad_node_mask[:compare_pad_start_idx])
        assert not torch.any(padded.pad_node_mask[compare_pad_start_idx:])

    for attr in _generate_homodata_node_attrs(original):
        if attr in exclude_keys:
            assert attr not in padded.keys()
            continue

        assert attr in padded.keys()

        if not isinstance(padded[attr], torch.Tensor):
            continue

        assert padded[attr].shape[0] == max_num_nodes

        # Check values in padded area.
        pad_value = node_pad_value.get_value(
            None, attr) if node_pad_value is not None else 0.0
        assert all(
            i == pad_value
            for i in torch.flatten(padded[attr][compare_pad_start_idx:]))

        # Check values in non-padded area.
        assert torch.equal(original[attr],
                           padded[attr][:compare_pad_start_idx])


def _check_homo_data_edges(
    original: Data,
    padded: Data,
    max_num_edges: Optional[int] = None,
    edge_pad_value: Optional[Padding] = None,
    is_mask_available: bool = False,
    exclude_keys: Optional[List[str]] = None,
):
    # Check edge index attribute.
    if max_num_edges is None:
        max_num_edges = padded.num_nodes**2
    assert padded.num_edges == max_num_edges
    assert padded.edge_index.shape[1] == max_num_edges
    assert padded.edge_index.shape[1] == max_num_edges

    compare_pad_start_idx = original.num_edges
    expected_node = original.num_nodes

    # Check values in padded area.
    assert all(
        padded.edge_index[0, i] == padded.edge_index[1, i] == expected_node
        for i in range(compare_pad_start_idx, max_num_edges))
    # Check values in non-padded area.
    assert torch.equal(original.edge_index,
                       padded.edge_index[:, :compare_pad_start_idx])

    if is_mask_available:
        assert padded.pad_edge_mask.numel() == padded.num_edges
        assert torch.all(padded.pad_edge_mask[:compare_pad_start_idx])
        assert not torch.any(padded.pad_edge_mask[compare_pad_start_idx:])

    # Check other attributes.
    for attr in _generate_homodata_edge_attrs(original):
        if attr == 'edge_index':
            continue
        if attr in exclude_keys:
            assert attr not in padded.keys()
            continue

        assert attr in padded.keys()

        if not isinstance(padded[attr], torch.Tensor):
            continue

        assert padded[attr].shape[0] == max_num_edges

        # Check values in padded area.
        pad_value = edge_pad_value.get_value(
            None, attr) if edge_pad_value is not None else 0.0
        assert all(
            i == pad_value
            for i in torch.flatten(padded[attr][compare_pad_start_idx:, :]))

        # Check values in non-padded area.
        assert torch.equal(original[attr],
                           padded[attr][:compare_pad_start_idx, :])


def _check_hetero_data_nodes(
    original: HeteroData,
    padded: HeteroData,
    max_num_nodes: Union[int, Dict[NodeType, int]],
    node_pad_value: Optional[Padding] = None,
    is_mask_available: bool = False,
    exclude_keys: Optional[List[str]] = None,
):
    if is_mask_available:
        for store in padded.node_stores:
            assert 'pad_node_mask' in store

    expected_nodes = max_num_nodes

    for node_type, attr in _generate_heterodata_nodes(original):
        if attr in exclude_keys:
            assert attr not in padded[node_type].keys()
            continue

        assert attr in padded[node_type].keys()

        if not isinstance(padded[node_type][attr], torch.Tensor):
            continue

        compare_pad_start_idx = original[node_type].num_nodes
        padded_tensor = padded[node_type][attr]

        if attr == 'pad_node_mask':
            assert padded_tensor.numel() == padded[node_type].num_nodes
            assert torch.all(padded_tensor[:compare_pad_start_idx])
            assert not torch.any(padded_tensor[compare_pad_start_idx:])
            continue

        original_tensor = original[node_type][attr]

        # Check the number of nodes.
        if isinstance(max_num_nodes, dict):
            expected_nodes = max_num_nodes[node_type]

        assert padded_tensor.shape[0] == expected_nodes

        compare_pad_start_idx = original_tensor.shape[0]
        pad_value = node_pad_value.get_value(
            node_type, attr) if node_pad_value is not None else 0.0
        assert all(
            i == pad_value
            for i in torch.flatten(padded_tensor[compare_pad_start_idx:]))
        # Compare non-padded area with the original.
        assert torch.equal(original_tensor,
                           padded_tensor[:compare_pad_start_idx])


def _check_hetero_data_edges(
    original: HeteroData,
    padded: HeteroData,
    max_num_edges: Optional[Union[int, Dict[EdgeType, int]]] = None,
    edge_pad_value: Optional[Padding] = None,
    is_mask_available: bool = False,
    exclude_keys: Optional[List[str]] = None,
):
    if is_mask_available:
        for store in padded.edge_stores:
            assert 'pad_edge_mask' in store

    for edge_type, attr in _generate_heterodata_edges(padded):
        if attr in exclude_keys:
            assert attr not in padded[edge_type].keys()
            continue

        assert attr in padded[edge_type].keys()

        if not isinstance(padded[edge_type][attr], torch.Tensor):
            continue

        compare_pad_start_idx = original[edge_type].num_edges
        padded_tensor = padded[edge_type][attr]

        if attr == 'pad_edge_mask':
            assert padded_tensor.numel() == padded[edge_type].num_edges
            assert torch.all(padded_tensor[:compare_pad_start_idx])
            assert not torch.any(padded_tensor[compare_pad_start_idx:])
            continue

        original_tensor = original[edge_type][attr]

        if isinstance(max_num_edges, numbers.Number):
            expected_num_edges = max_num_edges
        elif max_num_edges is None or edge_type not in max_num_edges.keys():
            v1, _, v2 = edge_type
            expected_num_edges = padded[v1].num_nodes * padded[v2].num_nodes
        else:
            expected_num_edges = max_num_edges[edge_type]

        if attr == 'edge_index':
            # Check the number of edges.
            assert padded_tensor.shape[1] == expected_num_edges

            # Check padded area values.
            src_nodes = original[edge_type[0]].num_nodes
            assert all(
                i == src_nodes
                for i in torch.flatten(padded_tensor[0,
                                                     compare_pad_start_idx:]))
            dst_nodes = original[edge_type[2]].num_nodes
            assert all(
                i == dst_nodes
                for i in torch.flatten(padded_tensor[1,
                                                     compare_pad_start_idx:]))

            # Compare non-padded area with the original.
            assert torch.equal(original_tensor,
                               padded_tensor[:, :compare_pad_start_idx])
        else:
            # Check padded area size.
            assert padded_tensor.shape[0] == expected_num_edges

            # Check padded area values.
            pad_value = edge_pad_value.get_value(
                edge_type, attr) if edge_pad_value is not None else 0.0
            assert all(i == pad_value for i in torch.flatten(padded_tensor[
                compare_pad_start_idx:, :]))

            # Compare non-padded area with the original.
            assert torch.equal(original_tensor,
                               padded_tensor[:compare_pad_start_idx, :])


def _check_data(
    original: Union[Data, HeteroData],
    padded: Union[Data, HeteroData],
    max_num_nodes: Union[int, Dict[NodeType, int]],
    max_num_edges: Optional[Union[int, Dict[EdgeType, int]]] = None,
    node_pad_value: Optional[Union[Padding, int, float]] = None,
    edge_pad_value: Optional[Union[Padding, int, float]] = None,
    is_mask_available: bool = False,
    exclude_keys: Optional[List[str]] = None,
):

    if not isinstance(node_pad_value, Padding) and node_pad_value is not None:
        node_pad_value = UniformPadding(node_pad_value)
    if not isinstance(edge_pad_value, Padding) and edge_pad_value is not None:
        edge_pad_value = UniformPadding(edge_pad_value)

    if is_mask_available is None:
        is_mask_available = False

    if exclude_keys is None:
        exclude_keys = []

    if isinstance(original, Data):
        _check_homo_data_nodes(original, padded, max_num_nodes, node_pad_value,
                               is_mask_available, exclude_keys)
        _check_homo_data_edges(original, padded, max_num_edges, edge_pad_value,
                               is_mask_available, exclude_keys)
    else:
        _check_hetero_data_nodes(original, padded, max_num_nodes,
                                 node_pad_value, is_mask_available,
                                 exclude_keys)
        _check_hetero_data_edges(original, padded, max_num_edges,
                                 edge_pad_value, is_mask_available,
                                 exclude_keys)


def test_pad_repr():
    pad_str = 'Pad(max_num_nodes=10, max_num_edges=15, ' \
        'node_pad_value=UniformPadding(value=3.0), ' \
        'edge_pad_value=UniformPadding(value=1.5))'
    assert str(eval(pad_str)) == pad_str


@pytest.mark.parametrize('data', [fake_data(), fake_hetero_data()])
@pytest.mark.parametrize('num_nodes', [32, 64])
@pytest.mark.parametrize('add_pad_mask', [True, False])
def test_pad_auto_edges(data, num_nodes, add_pad_mask):
    original = data
    data = deepcopy(data)
    transform = Pad(max_num_nodes=num_nodes, add_pad_mask=add_pad_mask)

    padded = transform(data)
    _check_data(original, padded, num_nodes, is_mask_available=add_pad_mask)


@pytest.mark.parametrize('num_nodes', [32, 64])
@pytest.mark.parametrize('num_edges', [300, 411])
@pytest.mark.parametrize('add_pad_mask', [True, False])
def test_pad_data_explicit_edges(num_nodes, num_edges, add_pad_mask):
    data = fake_data()
    original = deepcopy(data)
    transform = Pad(max_num_nodes=num_nodes, max_num_edges=num_edges,
                    add_pad_mask=add_pad_mask)

    padded = transform(data)
    _check_data(original, padded, num_nodes, num_edges,
                is_mask_available=add_pad_mask)


@pytest.mark.parametrize('num_nodes', [32, {'v0': 64, 'v1': 36}])
@pytest.mark.parametrize('num_edges', [300, {('v0', 'e0', 'v1'): 397}])
@pytest.mark.parametrize('add_pad_mask', [True, False])
def test_pad_heterodata_explicit_edges(num_nodes, num_edges, add_pad_mask):
    data = fake_hetero_data()
    original = deepcopy(data)
    transform = Pad(max_num_nodes=num_nodes, max_num_edges=num_edges,
                    add_pad_mask=add_pad_mask)

    padded = transform(data)
    _check_data(original, padded, num_nodes, num_edges,
                is_mask_available=add_pad_mask)


@pytest.mark.parametrize('node_pad_value', [10, AttrNamePadding({'x': 3.0})])
@pytest.mark.parametrize('edge_pad_value',
                         [11, AttrNamePadding({'edge_attr': 2.0})])
def test_pad_data_pad_values(node_pad_value, edge_pad_value):
    data = fake_data()
    original = deepcopy(data)
    num_nodes = 32
    transform = Pad(max_num_nodes=num_nodes, node_pad_value=node_pad_value,
                    edge_pad_value=edge_pad_value)
    padded = transform(data)
    _check_data(original, padded, num_nodes, node_pad_value=node_pad_value,
                edge_pad_value=edge_pad_value)


@pytest.mark.parametrize('node_pad_value', [
    UniformPadding(12),
    AttrNamePadding({'x': 0}),
    NodeTypePadding({
        'v0': UniformPadding(12),
        'v1': AttrNamePadding({'x': 7})
    })
])
@pytest.mark.parametrize('edge_pad_value', [
    UniformPadding(13),
    EdgeTypePadding({
        ('v0', 'e0', 'v1'):
        UniformPadding(13),
        ('v1', 'e0', 'v0'):
        AttrNamePadding({'edge_attr': UniformPadding(-1.0)})
    })
])
def test_pad_heterodata_pad_values(node_pad_value, edge_pad_value):
    data = fake_hetero_data()
    original = deepcopy(data)
    num_nodes = 32
    transform = Pad(max_num_nodes=num_nodes, node_pad_value=node_pad_value,
                    edge_pad_value=edge_pad_value)

    padded = transform(data)
    _check_data(original, padded, num_nodes, node_pad_value=node_pad_value,
                edge_pad_value=edge_pad_value)


@pytest.mark.parametrize('data', [fake_data(), fake_hetero_data()])
@pytest.mark.parametrize('add_pad_mask', [True, False])
@pytest.mark.parametrize('exclude_keys', [
    ['y'],
    ['edge_attr'],
    ['y', 'edge_attr'],
])
def test_pad_data_exclude_keys(data, add_pad_mask, exclude_keys):
    original = data
    data = deepcopy(data)
    num_nodes = 32
    transform = Pad(max_num_nodes=num_nodes, add_pad_mask=add_pad_mask,
                    exclude_keys=exclude_keys)

    padded = transform(data)
    _check_data(original, padded, num_nodes, is_mask_available=add_pad_mask,
                exclude_keys=exclude_keys)


@pytest.mark.parametrize('data', [fake_data(), fake_hetero_data(node_types=1)])
def test_pad_invalid_max_num_nodes(data):
    transform = Pad(max_num_nodes=data.num_nodes - 1)

    with pytest.raises(AssertionError,
                       match='The number of nodes after padding'):
        transform(data)


@pytest.mark.parametrize(
    'data',
    [fake_data(), fake_hetero_data(node_types=1, edge_types=1)])
def test_pad_invalid_max_num_edges(data):
    transform = Pad(max_num_nodes=data.num_nodes + 10,
                    max_num_edges=data.num_edges - 1)

    with pytest.raises(AssertionError,
                       match='The number of edges after padding'):
        transform(data)


def test_pad_num_nodes_not_complete():
    data = fake_hetero_data(node_types=2, edge_types=1)
    transform = Pad(max_num_nodes={'v0': 100})

    with pytest.raises(AssertionError, match='The number of v1 nodes'):
        transform(data)


def test_pad_invalid_padding_type():
    with pytest.raises(ValueError, match="to be an integer or float"):
        Pad(max_num_nodes=100, node_pad_value='somestring')
    with pytest.raises(ValueError, match="to be an integer or float"):
        Pad(max_num_nodes=100, edge_pad_value='somestring')


def test_pad_data_non_tensor_attr():
    data = deepcopy(fake_data())
    batch_size = 13
    data.batch_size = batch_size

    transform = Pad(max_num_nodes=100)
    padded = transform(data)
    assert padded.batch_size == batch_size

    exclude_transform = Pad(max_num_nodes=101, exclude_keys=('batch_size', ))
    padded = exclude_transform(data)
    assert 'batch_size' not in padded.keys()


@pytest.mark.parametrize('mask_pad_value', [True, False])
def test_pad_node_additional_attr_mask(mask_pad_value):
    data = fake_data()
    mask = torch.randn(data.num_nodes) > 0
    mask_names = ['train_mask', 'test_mask', 'val_mask']
    for mask_name in mask_names:
        setattr(data, mask_name, mask)
    padding_num = 20

    max_num_nodes = int(data.num_nodes) + padding_num
    max_num_edges = data.num_edges + padding_num

    transform = Pad(max_num_nodes, max_num_edges, node_pad_value=0.1,
                    mask_pad_value=mask_pad_value)
    padded = transform(data)
    padded_masks = [getattr(padded, mask_name) for mask_name in mask_names]

    for padded_mask in padded_masks:
        assert padded_mask.ndim == 1
        assert padded_mask.size()[0] == max_num_nodes
        assert torch.all(padded_mask[-padding_num:] == mask_pad_value)


def test_uniform_padding():
    pad_val = 10.0
    p = UniformPadding(pad_val)
    assert p.get_value() == pad_val
    assert p.get_value("v1", "x") == pad_val

    p = UniformPadding()
    assert p.get_value() == 0.0

    with pytest.raises(ValueError, match="to be an integer or float"):
        UniformPadding('')


def test_attr_name_padding():
    x_val = 10.0
    y_val = 15.0
    default = 3.0
    padding_dict = {'x': x_val, 'y': UniformPadding(y_val)}
    padding = AttrNamePadding(padding_dict, default=default)

    assert padding.get_value(attr_name='x') == x_val
    assert padding.get_value('v1', 'x') == x_val
    assert padding.get_value(attr_name='y') == y_val
    assert padding.get_value('v1', 'y') == y_val
    assert padding.get_value(attr_name='x2') == default

    padding = AttrNamePadding({})
    assert padding.get_value(attr_name='x') == 0.0


def test_attr_name_padding_invalid():
    with pytest.raises(ValueError, match="to be a dictionary"):
        AttrNamePadding(10.0)

    with pytest.raises(ValueError, match="to be a string"):
        AttrNamePadding({10: 10.0})

    with pytest.raises(ValueError, match="to be of type"):
        AttrNamePadding({"x": {}})

    with pytest.raises(ValueError, match="to be of type"):
        AttrNamePadding({"x": {}})

    node_type_padding = NodeTypePadding({"x": 10.0})
    with pytest.raises(ValueError, match="to be of type"):
        AttrNamePadding({'x': node_type_padding})


@pytest.mark.parametrize('store_type', ['node', 'edge'])
def test_node_edge_type_padding(store_type):
    if store_type == "node":
        stores = ['v1', 'v2', 'v3', 'v4']
        padding_cls = NodeTypePadding
    else:
        stores = [('v1', 'e1', 'v1'), ('v1', 'e2', 'v1'), ('v1', 'e1', 'v2'),
                  ('v2', 'e1', 'v1')]
        padding_cls = EdgeTypePadding

    s0_default = 3.0
    s0_padding_dict = {'x': 10.0, 'y': -12.0}
    s0_padding = AttrNamePadding(s0_padding_dict, s0_default)
    s1_default = 0.1
    s1_padding_dict = {'y': 0.0, 'p': 13.0}
    s1_padding = AttrNamePadding(s1_padding_dict, s1_default)

    s2_default = 7.5
    store_default = -11.0
    padding_dict = {
        stores[0]: s0_padding,
        stores[1]: s1_padding,
        stores[2]: s2_default
    }
    padding = padding_cls(padding_dict, store_default)

    assert padding.get_value(stores[0], 'x') == s0_padding_dict['x']
    assert padding.get_value(stores[0], 'y') == s0_padding_dict['y']
    assert padding.get_value(stores[0], 'p') == s0_default
    assert padding.get_value(stores[0], 'z') == s0_default

    assert padding.get_value(stores[1], 'x') == s1_default
    assert padding.get_value(stores[1], 'y') == s1_padding_dict['y']
    assert padding.get_value(stores[1], 'p') == s1_padding_dict['p']
    assert padding.get_value(stores[1], 'z') == s1_default

    assert padding.get_value(stores[2], 'x') == s2_default
    assert padding.get_value(stores[2], 'z') == s2_default

    assert padding.get_value(stores[3], 'x') == store_default


def test_edge_padding_invalid():
    with pytest.raises(ValueError, match="to be a tuple"):
        EdgeTypePadding({'v1': 10.0})

    with pytest.raises(ValueError, match="got 1"):
        EdgeTypePadding({('v1', ): 10.0})

    with pytest.raises(ValueError, match="got 2"):
        EdgeTypePadding({('v1', 'v2'): 10.0})

    with pytest.raises(ValueError, match="got 4"):
        EdgeTypePadding({('v1', 'e2', 'v1', 'v2'): 10.0})

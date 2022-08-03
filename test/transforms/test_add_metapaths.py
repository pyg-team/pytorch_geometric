import copy

import torch
from torch import tensor

from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths


def test_add_metapaths():
    data = HeteroData()
    data['p'].x = torch.ones(5)
    data['a'].x = torch.ones(6)
    data['c'].x = torch.ones(3)
    data['p', 'p'].edge_index = tensor([[0, 1, 2, 3], [1, 2, 4, 2]])
    data['p', 'a'].edge_index = tensor([[0, 1, 2, 3, 4], [2, 2, 5, 2, 5]])
    data['a', 'p'].edge_index = data['p', 'a'].edge_index.flip([0])
    data['c', 'p'].edge_index = tensor([[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]])
    data['p', 'c'].edge_index = data['c', 'p'].edge_index.flip([0])

    # Test transform options:
    metapaths = [[('p', 'c'), ('c', 'p')]]

    transform = AddMetaPaths(metapaths)
    assert str(transform) == 'AddMetaPaths()'
    meta1 = transform(copy.copy(data))

    transform = AddMetaPaths(metapaths, drop_orig_edges=True)
    assert str(transform) == 'AddMetaPaths()'
    meta2 = transform(copy.copy(data))

    transform = AddMetaPaths(metapaths, drop_orig_edges=True,
                             keep_same_node_type=True)
    assert str(transform) == 'AddMetaPaths()'
    meta3 = transform(copy.copy(data))

    transform = AddMetaPaths(metapaths, drop_orig_edges=True,
                             keep_same_node_type=True,
                             drop_unconnected_nodes=True)
    assert str(transform) == 'AddMetaPaths()'
    meta4 = transform(copy.copy(data))

    assert meta1['metapath_0'].edge_index.size() == (2, 9)
    assert meta2['metapath_0'].edge_index.size() == (2, 9)
    assert meta3['metapath_0'].edge_index.size() == (2, 9)
    assert meta4['metapath_0'].edge_index.size() == (2, 9)

    assert all([i in meta1.edge_types for i in data.edge_types])
    assert meta2.edge_types == [('p', 'metapath_0', 'p')]
    assert meta3.edge_types == [('p', 'to', 'p'), ('p', 'metapath_0', 'p')]
    assert meta4.edge_types == [('p', 'to', 'p'), ('p', 'metapath_0', 'p')]

    assert meta3.node_types == ['p', 'a', 'c']
    assert meta4.node_types == ['p']

    # Test 4-hop metapath:
    metapaths = [
        [('a', 'p'), ('p', 'c')],
        [('a', 'p'), ('p', 'c'), ('c', 'p'), ('p', 'a')],
    ]
    transform = AddMetaPaths(metapaths)
    meta = transform(copy.copy(data))
    new_edge_types = [('a', 'metapath_0', 'c'), ('a', 'metapath_1', 'a')]
    assert meta['metapath_0'].edge_index.size() == (2, 4)
    assert meta['metapath_1'].edge_index.size() == (2, 4)

    # Test `metapath_dict` information:
    assert list(meta.metapath_dict.values()) == metapaths
    assert list(meta.metapath_dict.keys()) == new_edge_types


def test_add_metapaths_max_sample():
    torch.manual_seed(12345)

    data = HeteroData()
    data['p'].x = torch.ones(5)
    data['a'].x = torch.ones(6)
    data['c'].x = torch.ones(3)
    data['p', 'p'].edge_index = tensor([[0, 1, 2, 3], [1, 2, 4, 2]])
    data['p', 'a'].edge_index = tensor([[0, 1, 2, 3, 4], [2, 2, 5, 2, 5]])
    data['a', 'p'].edge_index = data['p', 'a'].edge_index.flip([0])
    data['c', 'p'].edge_index = tensor([[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]])
    data['p', 'c'].edge_index = data['c', 'p'].edge_index.flip([0])

    metapaths = [[('p', 'c'), ('c', 'p')]]
    transform = AddMetaPaths(metapaths, max_sample=1)

    meta = transform(data)
    assert meta['metapath_0'].edge_index.size(1) < 9


def test_add_weighted_metapaths():
    torch.manual_seed(12345)

    data = HeteroData()
    data['a'].num_nodes = 2
    data['b'].num_nodes = 3
    data['c'].num_nodes = 2
    data['d'].num_nodes = 2
    data['a', 'b'].edge_index = tensor([[0, 1, 1], [0, 1, 2]])
    data['b', 'a'].edge_index = data['a', 'b'].edge_index.flip([0])
    data['b', 'c'].edge_index = tensor([[0, 1, 2], [0, 1, 1]])
    data['c', 'b'].edge_index = data['b', 'c'].edge_index.flip([0])
    data['c', 'd'].edge_index = tensor([[0, 1], [0, 0]])
    data['d', 'c'].edge_index = data['c', 'd'].edge_index.flip([0])

    metapaths = [
        [('a', 'b'), ('b', 'c')],
        [('a', 'b'), ('b', 'c'), ('c', 'd')],
        [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'c'), ('c', 'b'),
         ('b', 'a')],
    ]
    transform = AddMetaPaths(metapaths, weighted=True)
    metapath_data = transform(copy.copy(data))

    # Make sure manually added metapaths compute the correct number of edges
    assert metapath_data['a', 'c'].edge_weight.tolist() == [1, 2]
    assert metapath_data['a', 'd'].edge_weight.tolist() == [1, 2]
    assert metapath_data['a', 'a'].edge_weight.tolist() == [1, 2, 2, 4]

    # Compute intra-table metapaths efficiently
    metapaths = [[('a', 'b'), ('b', 'c'), ('c', 'd')]]
    metapath_data = AddMetaPaths(metapaths, weighted=True)(copy.copy(data))
    metapath_data['d',
                  'a'].edge_index = metapath_data['a',
                                                  'd'].edge_index.flip([0])
    metapath_data['d', 'a'].edge_weight = metapath_data['a', 'd'].edge_weight
    metapaths = [[('a', 'd'), ('d', 'a')]]
    metapath_data = AddMetaPaths(metapaths, weighted=True)(metapath_data)
    del metapath_data['a', 'd']
    del metapath_data['d', 'a']
    assert metapath_data['a', 'a'].edge_weight.tolist() == [1, 2, 2, 4]

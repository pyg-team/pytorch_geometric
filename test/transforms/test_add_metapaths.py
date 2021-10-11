import torch
from copy import deepcopy

from torch_geometric.transforms import AddMetaPaths
from torch_geometric.data import HeteroData


def test_add_metapaths():

    dblp = HeteroData()
    dblp['paper'].x = torch.ones(5)
    dblp['author'].x = torch.ones(6)
    dblp['conference'].x = torch.ones(3)
    dblp['paper', 'cites', 'paper'].edge_index = torch.tensor([[0, 1, 2, 3],
                                                               [1, 2, 4, 2]])
    dblp['paper', 'author'].edge_index = torch.tensor([[0, 1, 2, 3, 4],
                                                       [2, 2, 5, 2, 5]])
    dblp['author', 'paper'].edge_index = dblp['paper',
                                              'author'].edge_index[[1, 0]]
    dblp['conference', 'paper'].edge_index = torch.tensor([[0, 0, 1, 2, 2],
                                                           [0, 1, 2, 3, 4]])
    dblp['paper', 'conference'].edge_index = dblp['conference',
                                                  'paper'].edge_index[[1, 0]]

    # test transform options.
    orig_edge_type = deepcopy(dblp.edge_types)
    metapaths = [[('paper', 'to', 'conference'),
                  ('conference', 'to', 'paper')]]
    meta1 = AddMetaPaths(metapaths)(dblp.clone())
    meta2 = AddMetaPaths(metapaths, drop_orig_edges=True)(dblp.clone())
    meta3 = AddMetaPaths(metapaths, drop_orig_edges=True,
                         keep_same_node_type=True)(dblp.clone())

    assert meta1['paper', 'metapath_0', 'paper'].edge_index.shape[-1] == 9
    assert meta2['paper', 'metapath_0', 'paper'].edge_index.shape[-1] == 9
    assert meta3['paper', 'metapath_0', 'paper'].edge_index.shape[-1] == 9

    assert all([i in meta1.edge_types for i in orig_edge_type])
    assert meta2.edge_types == [('paper', 'metapath_0', 'paper')]
    assert meta3.edge_types == [('paper', 'cites', 'paper'),
                                ('paper', 'metapath_0', 'paper')]

    # test 4-hop metapath
    metapaths = [[('author', 'to', 'paper'), ('paper', 'to', 'conference')],
                 [('author', 'to', 'paper'), ('paper', 'to', 'conference'),
                  ('conference', 'to', 'paper'), ('paper', 'to', 'author')]]
    meta1 = AddMetaPaths(metapaths)(dblp.clone())
    assert meta1['author', 'metapath_0',
                 'conference'].edge_index.shape[-1] == 4
    assert meta1['author', 'metapath_1', 'author'].edge_index.shape[-1] == 4
    assert meta1.metapaths == metapaths

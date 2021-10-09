import torch
from copy import deepcopy

from torch_geometric.transforms import AddMetaPaths
from torch_geometric.data import HeteroData


def test_add_metapaths():

    imdb = HeteroData()
    imdb['movie'].x = torch.ones(5)
    imdb['person'].x = torch.ones(5)
    imdb['year'].x = torch.ones(5)
    imdb['movie', 'sequel', 'movie'].edge_index = torch.tensor([[0, 1, 2],
                                                                [1, 2, 3]])
    imdb['movie', 'actor', 'person'].edge_index = torch.tensor([[0, 1, 2],
                                                                [2, 2, 4]])
    imdb['person', 'acted', 'movie'].edge_index = torch.tensor([[2, 2, 4],
                                                                [0, 1, 2]])
    imdb['movie', 'director', 'person'].edge_index = torch.tensor([[0, 1, 2],
                                                                   [2, 2, 4]])
    imdb['movie', 'released', 'year'].edge_index = torch.tensor([[0, 1, 2],
                                                                 [2, 3, 4]])
    orig_edge_type = deepcopy(imdb.edge_types)

    meta1 = AddMetaPaths([[('movie', 'actor', 'person'),
                           ('person', 'acted', 'movie')]])(imdb.clone())
    meta2 = AddMetaPaths([[('movie', 'actor', 'person'),
                           ('person', 'acted', 'movie')]],
                         drop_orig_edges=True)(imdb.clone())
    meta3 = AddMetaPaths([[('movie', 'actor', 'person'),
                           ('person', 'acted', 'movie')]],
                         drop_orig_edges=True,
                         keep_same_node_type=True)(imdb.clone())

    assert meta1['movie', 'metapath_0', 'movie'].edge_index.shape[-1] == 5
    assert meta2['movie', 'metapath_0', 'movie'].edge_index.shape[-1] == 5
    assert meta3['movie', 'metapath_0', 'movie'].edge_index.shape[-1] == 5

    assert all([i in meta1.edge_types for i in orig_edge_type])
    assert meta2.edge_types == [('movie', 'metapath_0', 'movie')]
    assert all([
        i in [('movie', 'metapath_0', 'movie'), ('movie', 'sequel', 'movie')]
        for i in meta3.edge_types
    ])

    # test 4-hop metapath
    dblp = HeteroData()
    dblp['paper'].x = torch.ones(5)
    dblp['author'].x = torch.ones(6)
    dblp['conference'].x = torch.ones(3)
    dblp['paper', 'author'].edge_index = torch.tensor([[0, 1, 2, 3, 4],
                                                       [2, 2, 5, 2, 5]])
    dblp['author', 'paper'].edge_index = dblp['paper',
                                              'author'].edge_index[[1, 0]]
    dblp['conference', 'paper'].edge_index = torch.tensor([[0, 0, 1, 2, 2],
                                                           [0, 1, 2, 3, 4]])
    dblp['paper', 'conference'].edge_index = dblp['conference',
                                                  'paper'].edge_index[[1, 0]]

    metapaths = [[('author', 'to', 'paper'), ('paper', 'to', 'conference'),
                  ('conference', 'to', 'paper'), ('paper', 'to', 'author')]]
    meta1 = AddMetaPaths(metapaths)(dblp.clone())
    # assert meta1['author', 'metapath_0', 'author'].edge_index.shape[-1] == 10

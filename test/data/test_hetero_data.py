import torch
from torch_geometric.data import HeteroData

x_paper = torch.randn(10, 16)
x_author = torch.randn(5, 32)

idx_paper = torch.randint(x_paper.size(0), (100, ), dtype=torch.long)
idx_author = torch.randint(x_author.size(0), (100, ), dtype=torch.long)

edge_index_paper_paper = torch.stack([idx_paper[:50], idx_paper[:50]], dim=0)
edge_index_paper_author = torch.stack([idx_paper[:30], idx_author[:30]], dim=0)
edge_index_author_paper = torch.stack([idx_paper[:30], idx_author[:30]], dim=0)


def test_hetero_data_init():
    data = HeteroData()
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    assert len(data) == 2

    data = HeteroData(
        paper={'x': x_paper},
        author={'x': x_author},
        paper__paper={'edge_index': edge_index_paper_paper},
        paper__author={'edge_index': edge_index_paper_author},
        author__paper={'edge_index': edge_index_author_paper},
    )
    assert len(data) == 2

    data = HeteroData({
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
    assert edge_types == [('paper', '_', 'paper'), ('paper', '_', 'author'),
                          ('author', '_', 'paper')]

    x_dict = data.x_dict
    assert len(x_dict) == 2
    assert x_dict['paper'].tolist() == x_paper.tolist()
    assert x_dict['author'].tolist() == x_author.tolist()

    data.y = 0
    assert data['y'] == 0 and data.y == 0
    assert len(data) == 3
    assert sorted(data.keys) == ['edge_index', 'x', 'y']

    del data['paper', 'author']
    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', '_', 'paper'), ('author', '_', 'paper')]

    assert len(data.to_dict()) == 5
    assert len(data.to_namedtuple()) == 5
    assert data.to_namedtuple().y == 0
    assert len(data.to_namedtuple().paper) == 1

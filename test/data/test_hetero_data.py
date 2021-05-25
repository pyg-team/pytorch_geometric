import torch
from torch_geometric.data import HeteroData


def test_hetero_data():
    x1 = torch.randn(10, 16)
    x2 = torch.randn(5, 32)
    edge_index11 = torch.randint(10, (2, 30), dtype=torch.long)
    edge_index12 = torch.randint(10, (2, 30), dtype=torch.long)
    edge_index21 = torch.randint(10, (2, 30), dtype=torch.long)
    edge_index22 = torch.randint(10, (2, 10), dtype=torch.long)
    edge_index12 = edge_index12[:, edge_index12[1] < 5]
    edge_index21 = edge_index21[:, edge_index21[0] < 5]

    data = HeteroData()
    data['paper'].x = x1
    data['author'].x = x2
    data['paper', 'paper'].edge_index = edge_index11
    data['paper', 'author'].edge_index = edge_index12
    data['author', 'paper'].edge_index = edge_index21
    data['author', 'author'].edge_index = edge_index22
    assert len(data) == 2

    data = HeteroData(
        paper={'x': x1},
        author={'x': x2},
        paper__paper={'edge_index': edge_index11},
        paper__author={'edge_index': edge_index12},
        author__paper={'edge_index': edge_index21},
        author__author={'edge_index': edge_index22},
    )
    assert len(data) == 2

    data = HeteroData({
        'paper': {
            'x': x1
        },
        'author': {
            'x': x2
        },
        ('paper', 'paper'): {
            'edge_index': edge_index11
        },
        ('paper', 'author'): {
            'edge_index': edge_index12
        },
        ('author', 'paper'): {
            'edge_index': edge_index21
        },
        ('author', 'author'): {
            'edge_index': edge_index22
        }
    })
    assert len(data) == 2
    assert sorted(data.keys) == ['edge_index', 'x']
    assert 'x' in data and 'edge_index' in data
    assert data.num_nodes == 15
    assert data.num_edges == (edge_index11.numel() + edge_index12.numel() +
                              edge_index21.numel() + edge_index22.numel()) // 2

    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', '_', 'paper'), ('paper', '_', 'author'),
                          ('author', '_', 'paper'), ('author', '_', 'author')]

    x_dict = data.x_dict
    assert len(x_dict) == 2
    assert x_dict['paper'].tolist() == x1.tolist()
    assert x_dict['author'].tolist() == x2.tolist()

    data.y = 0
    assert data['y'] == 0 and data.y == 0
    assert len(data) == 3
    assert sorted(data.keys) == ['edge_index', 'x', 'y']

    del data['paper', 'author']
    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', '_', 'paper'), ('author', '_', 'paper'),
                          ('author', '_', 'author')]

    assert len(data.to_dict()) == 6
    assert len(data.to_namedtuple()) == 6
    assert data.to_namedtuple().y == 0
    assert len(data.to_namedtuple().paper) == 1

import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data


def test_compose():
    transform = T.Compose([T.Center(), T.AddSelfLoops()])
    assert str(transform) == ('Compose([\n'
                              '  Center(),\n'
                              '  AddSelfLoops()\n'
                              '])')

    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = Data(edge_index=edge_index, pos=pos)
    data = transform(data)
    assert len(data) == 2
    assert data.pos.tolist() == [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    assert data.edge_index.size() == (2, 7)


def test_compose_data_list():
    transform = T.Compose([T.Center(), T.AddSelfLoops()])

    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data_list = [Data(edge_index=edge_index, pos=pos) for _ in range(3)]
    data_list = transform(data_list)
    assert len(data_list) == 3
    for data in data_list:
        assert len(data) == 2
        assert data.pos.tolist() == [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
        assert data.edge_index.size() == (2, 7)


def test_compose_filters():
    filter_fn = T.ComposeFilters([
        lambda d: d.num_nodes > 2,
        lambda d: d.num_edges > 2,
    ])
    assert str(filter_fn)[:16] == 'ComposeFilters(['

    data1 = Data(x=torch.arange(3))
    assert not filter_fn(data1)

    data2 = Data(x=torch.arange(2), edge_index=torch.tensor([
        [0, 0, 1],
        [0, 1, 1],
    ]))
    assert not filter_fn(data2)

    data3 = Data(x=torch.arange(3), edge_index=torch.tensor([
        [0, 0, 1],
        [0, 1, 1],
    ]))
    assert filter_fn(data3)

    # Test tuple of data objects:
    assert filter_fn((data1, data2, data3)) is False

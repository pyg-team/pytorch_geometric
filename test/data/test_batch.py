import torch
import torch_geometric
from torch_geometric.data import Data, Batch


def test_batch():
    torch_geometric.set_debug(True)

    x1 = torch.tensor([1, 2, 3], dtype=torch.float)
    e1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    p1 = [torch.tensor([1, 0, 2]), torch.tensor([0, 2])]
    pf1 = [torch.tensor([False]), torch.tensor([True])]
    s1 = '1'
    x2 = torch.tensor([1, 2], dtype=torch.float)
    e2 = torch.tensor([[0, 1], [1, 0]])
    p2 = [torch.tensor([1, 0])]
    pf2 = [torch.tensor([True])]
    s2 = '2'

    data = Batch.from_data_list([
        Data(x1, e1, s=s1, paths_index=p1, paths_features=pf1),
        Data(x2, e2, s=s2, paths_index=p2, paths_features=pf2)
    ])

    assert data.__repr__() == (
        'Batch(batch=[5], edge_index=[2, 6], paths_features=[3], '
        'paths_index=[3], s=[2], x=[5])')
    assert len(data) == 6
    assert data.x.tolist() == [1, 2, 3, 1, 2]
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]
    assert [p.tolist() for p in data.paths_index] == [[1, 0, 2], [0, 2],
                                                      [4, 3]]
    assert data.s == ['1', '2']
    assert data.batch.tolist() == [0, 0, 0, 1, 1]
    assert data.num_graphs == 2

    data_list = data.to_data_list()
    assert len(data_list) == 2
    assert len(data_list[0]) == 5
    assert data_list[0].x.tolist() == [1, 2, 3]
    assert data_list[0].edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert [p.tolist() for p in data_list[0].paths_index] == [[1, 0, 2],
                                                              [0, 2]]
    assert data_list[0].s == '1'
    assert len(data_list[1]) == 5
    assert data_list[1].x.tolist() == [1, 2]
    assert data_list[1].edge_index.tolist() == [[0, 1], [1, 0]]
    assert [p.tolist() for p in data_list[1].paths_index] == [[1, 0]]
    assert data_list[1].s == '2'

    torch_geometric.set_debug(True)

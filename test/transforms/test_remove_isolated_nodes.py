import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RemoveIsolatedNodes


def test_remove_isolated_nodes():
    assert str(RemoveIsolatedNodes()) == 'RemoveIsolatedNodes()'

    data = Data()
    data.x = torch.arange(3)
    data.edge_index = torch.tensor([[0, 2], [2, 0]])
    data.edge_attr = torch.arange(2)

    data = RemoveIsolatedNodes()(data)

    assert len(data) == 3
    assert data.x.tolist() == [0, 2]
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]
    assert data.edge_attr.tolist() == [0, 1]


def test_remove_isolated_nodes_in_hetero_data():
    data = HeteroData()

    data['p'].x = torch.arange(6)
    data['a'].x = torch.arange(6)
    data['i'].num_nodes = 4

    # isolated paper nodes: {4}
    # isolated author nodes: {3, 4, 5}
    # isolated institution nodes: {0, 1, 2, 3}
    data['p', '1', 'p'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 3]])
    data['p', '2', 'a'].edge_index = torch.tensor([[1, 3, 5], [0, 1, 2]])
    data['p', '2', 'a'].edge_attr = torch.arange(3)
    data['p', '3', 'a'].edge_index = torch.tensor([[5], [2]])

    data = RemoveIsolatedNodes()(data)

    assert len(data) == 4
    assert data['p'].num_nodes == 5
    assert data['a'].num_nodes == 3
    assert data['i'].num_nodes == 0

    assert data['p'].x.tolist() == [0, 1, 2, 3, 5]
    assert data['a'].x.tolist() == [0, 1, 2]

    assert data['1'].edge_index.tolist() == [[0, 1, 2], [0, 1, 3]]
    assert data['2'].edge_index.tolist() == [[1, 3, 4], [0, 1, 2]]
    assert data['2'].edge_attr.tolist() == [0, 1, 2]
    assert data['3'].edge_index.tolist() == [[4], [2]]

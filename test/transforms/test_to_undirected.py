import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected


def test_to_undirected():
    assert str(ToUndirected()) == 'ToUndirected()'

    edge_index = torch.tensor([[2, 0, 2], [3, 1, 0]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 2, 1, 2, 0, 0])

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=4)
    data = ToUndirected()(data)
    assert len(data) == 4
    assert data.edge_index.tolist() == [[0, 0, 1, 2, 2, 3], [1, 2, 0, 0, 3, 2]]
    assert data.edge_weight.tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.num_nodes == 4


def test_to_undirected_with_duplicates():
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 2]])
    edge_weight = torch.ones(4)

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)
    data = ToUndirected()(data)
    assert len(data) == 3
    assert data.edge_index.tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 2, 1]]
    assert data.edge_weight.tolist() == [2, 2, 2, 1, 1]
    assert data.num_nodes == 3


def test_hetero_to_undirected():
    edge_index = torch.tensor([[2, 0], [3, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 1, 0, 0])

    data = HeteroData()
    data['v'].num_nodes = 4
    data['w'].num_nodes = 4
    data['v', 'v'].edge_index = edge_index
    data['v', 'v'].edge_weight = edge_weight
    data['v', 'v'].edge_attr = edge_attr
    data['v', 'w'].edge_index = edge_index
    data['v', 'w'].edge_weight = edge_weight
    data['v', 'w'].edge_attr = edge_attr

    from torch_geometric.transforms import ToUndirected

    assert not data.is_undirected()
    data = ToUndirected()(data)
    assert data.is_undirected()

    assert data['v', 'v'].edge_index.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert data['v', 'v'].edge_weight.tolist() == edge_weight[perm].tolist()
    assert data['v', 'v'].edge_attr.tolist() == edge_attr[perm].tolist()
    assert data['v', 'w'].edge_index.tolist() == edge_index.tolist()
    assert data['v', 'w'].edge_weight.tolist() == edge_weight.tolist()
    assert data['v', 'w'].edge_attr.tolist() == edge_attr.tolist()
    assert data['w', 'v'].edge_index.tolist() == [[3, 1], [2, 0]]
    assert data['w', 'v'].edge_weight.tolist() == edge_weight.tolist()
    assert data['w', 'v'].edge_attr.tolist() == edge_attr.tolist()


def test_hetero_to_undirected_self_relation_with_edge_label():
    edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
    edge_label_index = edge_index.clone()
    edge_label = torch.tensor([1, 0, 1])

    data = HeteroData()
    data['v'].num_nodes = 6
    data['v', 'v'].edge_index = edge_index
    data['v', 'v'].edge_label_index = edge_label_index
    data['v', 'v'].edge_label = edge_label

    data = ToUndirected()(data)

    # `edge_index` is made undirected (E -> 2 * E):
    assert data['v', 'v'].edge_index.size(1) == 6
    # Link-prediction supervision attributes are preserved unchanged on the
    # forward relation:
    assert data['v',
                'v'].edge_label_index.tolist() == edge_label_index.tolist()
    assert data['v', 'v'].edge_label.tolist() == edge_label.tolist()

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToSparseTensor


def test_to_sparse_tensor():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 0, 3, 2])

    transform = ToSparseTensor()
    assert str(transform) == ('ToSparseTensor(attr=edge_weight, '
                              'remove_edge_index=True, '
                              'fill_cache=True, '
                              'backend=torch_sparse)')
    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)
    data = transform(data)
    assert len(data) == 3
    assert data.adj_t.storage.row().tolist() == [0, 1, 1, 2]
    assert data.adj_t.storage.col().tolist() == [1, 0, 2, 1]
    assert data.adj_t.storage.value().tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.num_nodes == 3

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)
    transform = ToSparseTensor(backend='torch')
    assert str(transform) == ('ToSparseTensor(attr=edge_weight, '
                              'remove_edge_index=True, '
                              'fill_cache=True, '
                              'backend=torch)')
    data = transform(data)
    assert len(data) == 3
    assert data.adj_t.layout == torch.sparse_csr
    adj_coo = data.adj_t.to_sparse_coo()
    assert adj_coo._indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert adj_coo._values().tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.num_nodes == 3


def test_to_sparse_tensor_and_keep_edge_index():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 0, 3, 2])

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)
    data = ToSparseTensor(remove_edge_index=False, backend='torch')(data)
    assert len(data) == 5
    assert data.adj_t.layout == torch.sparse_csr
    adj_coo = data.adj_t.to_sparse_coo()
    assert adj_coo._indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert adj_coo._values().tolist() == edge_weight[perm].tolist()
    assert data.edge_index.tolist() == edge_index[:, perm].tolist()
    assert data.edge_weight.tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.num_nodes == 3

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)
    data = ToSparseTensor(remove_edge_index=False)(data)
    assert len(data) == 5
    assert data.adj_t.storage.row().tolist() == [0, 1, 1, 2]
    assert data.adj_t.storage.col().tolist() == [1, 0, 2, 1]
    assert data.adj_t.storage.value().tolist() == edge_weight[perm].tolist()
    assert data.edge_index.tolist() == edge_index[:, perm].tolist()
    assert data.edge_weight.tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.num_nodes == 3


def test_hetero_to_sparse_tensor():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = HeteroData()
    data['v'].num_nodes = 3
    data['w'].num_nodes = 3
    data['v', 'v'].edge_index = edge_index
    data['v', 'w'].edge_index = edge_index
    data = ToSparseTensor()(data)
    assert data['v', 'v'].adj_t.storage.row().tolist() == [0, 1, 1, 2]
    assert data['v', 'v'].adj_t.storage.col().tolist() == [1, 0, 2, 1]
    assert data['v', 'v'].adj_t.storage.value() is None
    assert data['v', 'w'].adj_t.storage.row().tolist() == [0, 1, 1, 2]
    assert data['v', 'w'].adj_t.storage.col().tolist() == [1, 0, 2, 1]
    assert data['v', 'w'].adj_t.storage.value() is None

    data = HeteroData()
    data['v'].num_nodes = 3
    data['w'].num_nodes = 3
    data['v', 'v'].edge_index = edge_index
    data['v', 'w'].edge_index = edge_index
    data = ToSparseTensor(backend='torch')(data)
    adj_coo = data['v', 'v'].adj_t.to_sparse_coo()
    assert adj_coo._indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert adj_coo._values().tolist() == [1., 1., 1., 1.]
    adj_coo = data['v', 'w'].adj_t.to_sparse_coo()
    assert adj_coo._indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert adj_coo._values().tolist() == [1., 1., 1., 1.]


def test_to_sparse_tensor_num_nodes_equals_num_edges():
    x = torch.arange(4)
    y = torch.arange(4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 0, 3, 2])

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, y=y)
    data = ToSparseTensor()(data)
    assert len(data) == 4
    assert data.x.tolist() == [0, 1, 2, 3]
    assert data.adj_t.storage.row().tolist() == [0, 1, 1, 2]
    assert data.adj_t.storage.col().tolist() == [1, 0, 2, 1]
    assert data.adj_t.storage.value().tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.y.tolist() == [0, 1, 2, 3]

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, y=y)
    data = ToSparseTensor(backend='torch')(data)
    assert len(data) == 4
    assert data.x.tolist() == [0, 1, 2, 3]
    adj_coo = data.adj_t.to_sparse_coo()
    assert adj_coo._indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert adj_coo._values().tolist() == edge_weight[perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()
    assert data.y.tolist() == [0, 1, 2, 3]

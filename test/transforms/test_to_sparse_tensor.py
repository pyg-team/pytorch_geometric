import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToSparseTensor


@pytest.mark.parametrize('layout', [None, torch.sparse_coo, torch.sparse_csr])
def test_to_sparse_tensor_basic(layout):
    transform = ToSparseTensor(layout=layout)
    assert str(transform) == (f'ToSparseTensor(attr=edge_weight, '
                              f'layout={layout})')

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 0, 3, 2])

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)
    data = transform(data)

    assert len(data) == 3
    assert data.num_nodes == 3
    assert torch.equal(data.edge_attr, edge_attr[perm])
    assert 'adj_t' in data

    if layout is None:  # `torch_sparse.SparseTensor`.
        row, col, value = data.adj_t.coo()
        assert row.tolist() == [0, 1, 1, 2]
        assert col.tolist() == [1, 0, 2, 1]
        assert torch.equal(value, edge_weight[perm])
    else:
        adj_t = data.adj_t
        assert adj_t.layout == layout
        if layout != torch.sparse_coo:
            adj_t = adj_t.to_sparse_coo()
        assert adj_t.indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
        assert torch.equal(adj_t.values(), edge_weight[perm])


def test_to_sparse_tensor_and_keep_edge_index():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    perm = torch.tensor([1, 0, 3, 2])

    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=3)
    data = ToSparseTensor(remove_edge_index=False)(data)

    assert len(data) == 5
    assert torch.equal(data.edge_index, edge_index[:, perm])
    assert torch.equal(data.edge_weight, edge_weight[perm])
    assert torch.equal(data.edge_attr, edge_attr[perm])


@pytest.mark.parametrize('layout', [None, torch.sparse_coo, torch.sparse_csr])
def test_hetero_to_sparse_tensor(layout):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = HeteroData()
    data['v'].num_nodes = 3
    data['w'].num_nodes = 3
    data['v', 'v'].edge_index = edge_index
    data['v', 'w'].edge_index = edge_index

    data = ToSparseTensor(layout=layout)(data)

    if layout is None:  # `torch_sparse.SparseTensor`.
        row, col, value = data['v', 'v'].adj_t.coo()
        assert row.tolist() == [0, 1, 1, 2]
        assert col.tolist() == [1, 0, 2, 1]
        assert value is None

        row, col, value = data['v', 'w'].adj_t.coo()
        assert row.tolist() == [0, 1, 1, 2]
        assert col.tolist() == [1, 0, 2, 1]
        assert value is None
    else:
        adj_t = data['v', 'v'].adj_t
        assert adj_t.layout == layout
        if layout != torch.sparse_coo:
            adj_t = adj_t.to_sparse_coo()
        assert adj_t.indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
        assert adj_t.values().tolist() == [1., 1., 1., 1.]

        adj_t = data['v', 'w'].adj_t
        assert adj_t.layout == layout
        if layout != torch.sparse_coo:
            adj_t = adj_t.to_sparse_coo()
        assert adj_t.indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
        assert adj_t.values().tolist() == [1., 1., 1., 1.]


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
    assert torch.equal(data.x, x)
    assert torch.equal(data.y, y)
    assert torch.equal(data.edge_attr, edge_attr[perm])

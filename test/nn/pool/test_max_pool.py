import torch

from torch_geometric.data import Batch
from torch_geometric.nn import max_pool, max_pool_neighbor_x, max_pool_x
from torch_geometric.testing import is_full_test


def test_max_pool_x():
    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    out = max_pool_x(cluster, x, batch)
    assert out[0].tolist() == [[5, 6], [7, 8], [11, 12]]
    assert out[1].tolist() == [0, 0, 1]

    if is_full_test():
        jit = torch.jit.script(max_pool_x)
        out = jit(cluster, x, batch)
        assert out[0].tolist() == [[5, 6], [7, 8], [11, 12]]
        assert out[1].tolist() == [0, 0, 1]

    out, _ = max_pool_x(cluster, x, batch, size=2)
    assert out.tolist() == [[5, 6], [7, 8], [11, 12], [0, 0]]

    batch_size = int(batch.max().item()) + 1
    out2, _ = max_pool_x(cluster, x, batch, batch_size=batch_size, size=2)
    assert torch.equal(out, out2)

    if is_full_test():
        jit = torch.jit.script(max_pool_x)
        out, _ = jit(cluster, x, batch, size=2)
        assert out.tolist() == [[5, 6], [7, 8], [11, 12], [0, 0]]

        out2, _ = jit(cluster, x, batch, batch_size=batch_size, size=2)
        assert torch.equal(out, out2)


def test_max_pool():
    cluster = torch.tensor([0, 1, 0, 1, 2, 2])
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    pos = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    edge_attr = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    data = Batch(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                 batch=batch)

    data = max_pool(cluster, data, transform=lambda x: x)

    assert data.x.tolist() == [[5, 6], [7, 8], [11, 12]]
    assert data.pos.tolist() == [[1, 1], [2, 2], [4.5, 4.5]]
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]
    assert data.edge_attr.tolist() == [4, 4]
    assert data.batch.tolist() == [0, 0, 1]


def test_max_pool_neighbor_x():
    x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    data = Batch(x=x, edge_index=edge_index, batch=batch)
    data = max_pool_neighbor_x(data)

    assert data.x.tolist() == [
        [7, 8],
        [7, 8],
        [7, 8],
        [7, 8],
        [11, 12],
        [11, 12],
    ]
    assert torch.equal(data.edge_index, edge_index)

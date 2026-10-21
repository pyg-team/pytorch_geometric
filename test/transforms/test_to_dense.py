import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.transforms import ToDense


def test_to_dense():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 4))
    pos = torch.randn((num_nodes, 3))
    y = torch.randint(0, 4, (num_nodes, ), dtype=torch.long)

    transform = ToDense()
    assert str(transform) == 'ToDense()'
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data = transform(data)
    assert len(data) == 5
    assert data.x.tolist() == x.tolist()
    assert data.pos.tolist() == pos.tolist()
    assert data.y.tolist() == y.tolist()
    assert data.adj.size() == (num_nodes, num_nodes)
    assert data.adj.tolist() == [
        [0, 1, 2, 3],
        [4, 0, 0, 0],
        [5, 0, 0, 0],
        [6, 0, 0, 0],
    ]
    assert data.mask.tolist() == [1, 1, 1, 1]

    transform = ToDense(num_nodes=5)
    assert str(transform) == 'ToDense(num_nodes=5)'
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data = transform(data)
    assert len(data) == 5
    assert data.x.size() == (5, 4)
    assert data.x[:4].tolist() == x.tolist()
    assert data.x[4].tolist() == [0, 0, 0, 0]
    assert data.pos.size() == (5, 3)
    assert data.pos[:4].tolist() == pos.tolist()
    assert data.pos[4].tolist() == [0, 0, 0]
    assert data.y.size() == (5, )
    assert data.y[:4].tolist() == y.tolist()
    assert data.y[4].tolist() == 0
    assert data.adj.size() == (5, 5)
    assert data.adj.tolist() == [
        [0, 1, 2, 3, 0],
        [4, 0, 0, 0, 0],
        [5, 0, 0, 0, 0],
        [6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert data.mask.tolist() == [1, 1, 1, 1, 0]


@pytest.mark.parametrize('num_nodes', [None, 4])
def test_to_dense_scalar_graph_labels(num_nodes):
    edge_index = torch.tensor([[0, 1], [1, 0]])
    graphs = [
        Data(x=torch.ones(3, 2), edge_index=edge_index, y=torch.tensor(y))
        for y in [0, 1]
    ]
    dense_graphs = [ToDense(num_nodes)(data) for data in graphs]
    for original, dense in zip(graphs, dense_graphs):
        assert dense.y.dim() == 0
        assert torch.equal(dense.y, original.y)
        assert original.edge_index is not None
    batch = next(iter(DenseDataLoader(dense_graphs, batch_size=2)))
    assert batch.y.tolist() == [0, 1]
    expected_nodes = num_nodes or 3
    assert batch.adj.size() == (2, expected_nodes, expected_nodes)
    assert batch.mask.sum(dim=1).tolist() == [3, 3]
    logits = torch.tensor([[2., 0.], [0., 2.]])
    loss = torch.nn.functional.cross_entropy(logits, batch.y)
    assert torch.isfinite(loss)

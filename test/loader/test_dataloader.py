import multiprocessing
import sys

import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

with_mp = sys.platform not in ['win32']
num_workers_list = [0, 2] if with_mp else [0]

if sys.platform == 'darwin':
    multiprocessing.set_start_method('spawn')


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.mark.parametrize('num_workers', num_workers_list)
def test_dataloader(num_workers):
    x = torch.Tensor([[1], [1], [1]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])
    y = 2.
    z = torch.tensor(0.)
    name = 'data'

    data = Data(x=x, edge_index=edge_index, y=y, z=z, name=name)
    assert str(data) == (
        "Data(x=[3, 1], edge_index=[2, 4], y=2.0, z=0.0, name='data')")
    data.face = face

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        num_workers=num_workers)
    assert len(loader) == 2

    for batch in loader:
        assert len(batch) == 8
        assert batch.batch.tolist() == [0, 0, 0, 1, 1, 1]
        assert batch.ptr.tolist() == [0, 3, 6]
        assert batch.x.tolist() == [[1], [1], [1], [1], [1], [1]]
        assert batch.edge_index.tolist() == [[0, 1, 1, 2, 3, 4, 4, 5],
                                             [1, 0, 2, 1, 4, 3, 5, 4]]
        assert batch.y.tolist() == [2.0, 2.0]
        assert batch.z.tolist() == [0.0, 0.0]
        assert batch.name == ['data', 'data']
        assert batch.face.tolist() == [[0, 3], [1, 4], [2, 5]]

        for store in batch.stores:
            assert id(batch) == id(store._parent())

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        follow_batch=['edge_index'], num_workers=num_workers)
    assert len(loader) == 2

    for batch in loader:
        assert len(batch) == 10
        assert batch.edge_index_batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.mark.skipif(not with_mp, reason='Multi-processing not available')
def test_multiprocessing():
    queue = torch.multiprocessing.Manager().Queue()
    data = Data(x=torch.randn(5, 16))
    data_list = [data, data, data, data]
    loader = DataLoader(data_list, batch_size=2)
    for batch in loader:
        queue.put(batch)

    batch = queue.get()
    assert len(batch) == 3

    batch = queue.get()
    assert len(batch) == 3


def test_pin_memory():
    x = torch.randn(3, 16)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index)

    loader = DataLoader([data] * 4, batch_size=2, pin_memory=True)
    for batch in loader:
        assert batch.x.is_pinned() or not torch.cuda.is_available()
        assert batch.edge_index.is_pinned() or not torch.cuda.is_available()


@pytest.mark.parametrize('num_workers', num_workers_list)
def test_heterogeneous_dataloader(num_workers):
    data = HeteroData()
    data['p'].x = torch.randn(100, 128)
    data['a'].x = torch.randn(200, 128)
    data['p', 'a'].edge_index = get_edge_index(100, 200, 500)
    data['p'].edge_attr = torch.randn(500, 32)
    data['a', 'p'].edge_index = get_edge_index(200, 100, 400)
    data['a', 'p'].edge_attr = torch.randn(400, 32)

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        num_workers=num_workers)
    assert len(loader) == 2

    for batch in loader:
        assert len(batch) == 5
        assert batch.num_nodes == 600

        for store in batch.stores:
            assert id(batch) == id(store._parent())

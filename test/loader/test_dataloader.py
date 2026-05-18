import math
import multiprocessing
import statistics
import sys
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_geometric import EdgeIndex, Index
from torch_geometric.data import Data, HeteroData, OnDiskDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import _mps_freeze_at
from torch_geometric.testing import (
    get_random_edge_index,
    get_random_tensor_frame,
    onlyLinux,
    withDevice,
    withPackage,
)

with_mp = sys.platform not in ['win32']
num_workers_list = [0, 2] if with_mp else [0]

if sys.platform == 'darwin':
    multiprocessing.set_start_method('spawn')


@withDevice
@pytest.mark.parametrize('num_workers', num_workers_list)
def test_dataloader(num_workers, device):
    if num_workers > 0 and device != torch.device('cpu'):
        return

    x = torch.tensor([[1.0], [1.0], [1.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])
    y = 2.
    z = torch.tensor(0.)
    name = 'data'

    data = Data(x=x, edge_index=edge_index, y=y, z=z, name=name).to(device)
    assert str(data) == ("Data(x=[3, 1], edge_index=[2, 4], y=2.0, z=0.0, "
                         "name='data')")
    data.face = face

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        num_workers=num_workers)
    assert len(loader) == 2

    for batch in loader:
        assert batch.x.device == device
        assert batch.edge_index.device == device
        assert batch.z.device == device
        assert batch.num_graphs == len(batch) == 2
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
                        follow_batch=['edge_index'], num_workers=num_workers,
                        collate_fn=None)
    assert len(loader) == 2

    for batch in loader:
        assert batch.num_graphs == len(batch) == 2
        assert batch.edge_index_batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


@onlyLinux
@pytest.mark.parametrize('num_workers', num_workers_list)
def test_dataloader_on_disk_dataset(tmp_path, num_workers):
    dataset = OnDiskDataset(tmp_path)
    data1 = Data(x=torch.randn(3, 8))
    data2 = Data(x=torch.randn(4, 8))
    dataset.extend([data1, data2])

    loader = DataLoader(dataset, batch_size=2, num_workers=num_workers)
    assert len(loader) == 1
    batch = next(iter(loader))
    assert batch.num_nodes == 7
    assert torch.equal(batch.x, torch.cat([data1.x, data2.x], dim=0))
    assert batch.batch.tolist() == [0, 0, 0, 1, 1, 1, 1]

    dataset.close()


def test_dataloader_fallbacks():
    # Test inputs of type List[torch.Tensor]:
    data_list = [torch.ones(3) for _ in range(4)]
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert torch.equal(batch, torch.ones(4, 3))

    # Test inputs of type List[float]:
    data_list = [1.0, 1.0, 1.0, 1.0]
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert torch.equal(batch, torch.ones(4))

    # Test inputs of type List[int]:
    data_list = [1, 1, 1, 1]
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert torch.equal(batch, torch.ones(4, dtype=torch.long))

    # Test inputs of type List[str]:
    data_list = ['test'] * 4
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert batch == data_list

    # Test inputs of type List[Mapping]:
    data_list = [{'x': torch.ones(3), 'y': 1}] * 4
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert torch.equal(batch['x'], torch.ones(4, 3))
    assert torch.equal(batch['y'], torch.ones(4, dtype=torch.long))

    # Test inputs of type List[Tuple]:
    DataTuple = namedtuple('DataTuple', 'x y')
    data_list = [DataTuple(0.0, 1)] * 4
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert torch.equal(batch.x, torch.zeros(4))
    assert torch.equal(batch[1], torch.ones(4, dtype=torch.long))

    # Test inputs of type List[Sequence]:
    data_list = [[0.0, 1]] * 4
    batch = next(iter(DataLoader(data_list, batch_size=4)))
    assert torch.equal(batch[0], torch.zeros(4))
    assert torch.equal(batch[1], torch.ones(4, dtype=torch.long))

    # Test that inputs of unsupported types raise an error:
    class DummyClass:
        pass

    with pytest.raises(TypeError):
        data_list = [DummyClass()] * 4
        next(iter(DataLoader(data_list, batch_size=4)))


@pytest.mark.skipif(not with_mp, reason='Multi-processing not available')
def test_multiprocessing():
    queue = torch.multiprocessing.Manager().Queue()
    data = Data(x=torch.randn(5, 16))
    data_list = [data, data, data, data]
    loader = DataLoader(data_list, batch_size=2)
    for batch in loader:
        queue.put(batch)

    batch = queue.get()
    assert batch.num_graphs == len(batch) == 2

    batch = queue.get()
    assert batch.num_graphs == len(batch) == 2


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
    data['p', 'a'].edge_index = get_random_edge_index(100, 200, 500)
    data['p'].edge_attr = torch.randn(500, 32)
    data['a', 'p'].edge_index = get_random_edge_index(200, 100, 400)
    data['a', 'p'].edge_attr = torch.randn(400, 32)

    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False,
                        num_workers=num_workers)
    assert len(loader) == 2

    for batch in loader:
        assert batch.num_graphs == len(batch) == 2
        assert batch.num_nodes == 600

        for store in batch.stores:
            assert id(batch) == id(store._parent())


@pytest.mark.parametrize('num_workers', num_workers_list)
def test_index_dataloader(num_workers):
    index1 = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    index2 = Index([0, 1, 1, 2, 2, 3], dim_size=4, is_sorted=True)

    data1 = Data(index=index1, num_nodes=3)
    data2 = Data(index=index2, num_nodes=4)

    loader = DataLoader(
        [data1, data2, data1, data2],
        batch_size=2,
        num_workers=num_workers,
    )
    assert len(loader) == 2

    for batch in loader:
        assert isinstance(batch.index, Index)
        assert batch.index.dtype == torch.long
        assert batch.index.dim_size == 7
        assert batch.index.is_sorted


@pytest.mark.parametrize('num_workers', num_workers_list)
@pytest.mark.parametrize('sort_order', [None, 'row', 'col'])
def test_edge_index_dataloader(num_workers, sort_order):
    if sort_order == 'col':
        edge_index = [[1, 0, 2, 1], [0, 1, 1, 2]]
    else:
        edge_index = [[0, 1, 1, 2], [1, 0, 2, 1]]

    edge_index = EdgeIndex(
        edge_index,
        sparse_size=(3, 3),
        sort_order=sort_order,
        is_undirected=True,
    )
    data = Data(edge_index=edge_index)
    assert data.num_nodes == 3

    loader = DataLoader(
        [data, data, data, data],
        batch_size=2,
        num_workers=num_workers,
    )
    assert len(loader) == 2

    for batch in loader:
        assert isinstance(batch.edge_index, EdgeIndex)
        assert batch.edge_index.dtype == torch.long
        assert batch.edge_index.sparse_size() == (6, 6)
        assert batch.edge_index.sort_order == sort_order
        assert batch.edge_index.is_undirected


@withPackage('torch_frame')
def test_dataloader_tensor_frame():
    tf = get_random_tensor_frame(num_rows=10)
    loader = DataLoader([tf, tf, tf, tf], batch_size=2, shuffle=False)
    assert len(loader) == 2

    for batch in loader:
        assert batch.num_rows == 20

    data = Data(tf=tf, edge_index=get_random_edge_index(10, 10, 20))
    loader = DataLoader([data, data, data, data], batch_size=2, shuffle=False)
    assert len(loader) == 2

    for batch in loader:
        assert batch.num_graphs == len(batch) == 2
        assert batch.num_nodes == 20
        assert batch.tf.num_rows == 20
        assert batch.edge_index.max() >= 10


def test_dataloader_sparse():
    adj_t = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        values=torch.randn(4),
        size=(3, 3),
    )
    data = Data(adj_t=adj_t)

    loader = DataLoader([data, data], batch_size=2)
    for batch in loader:
        assert batch.adj_t.size() == (6, 6)


if __name__ == '__main__':
    import argparse
    import time

    from torch_geometric.datasets import QM9

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    kwargs = dict(batch_size=128, shuffle=True, num_workers=args.num_workers)

    in_memory_dataset = QM9('/tmp/QM9')
    loader = DataLoader(in_memory_dataset, **kwargs)

    print('In-Memory Dataset:')
    for _ in range(2):
        print(f'Start loading {len(loader)} mini-batches ... ', end='')
        t = time.perf_counter()
        for _ in loader:
            pass
        print(f'Done! [{time.perf_counter() - t:.4f}s]')

    on_disk_dataset = in_memory_dataset.to_on_disk_dataset()
    loader = DataLoader(on_disk_dataset, **kwargs)

    print('On-Disk Dataset:')
    for _ in range(2):
        print(f'Start loading {len(loader)} mini-batches ... ', end='')
        t = time.perf_counter()
        for _ in loader:
            pass
        print(f'Done! [{time.perf_counter() - t:.4f}s]')

    on_disk_dataset.close()


def _make_dataset(n=20, node_range=(5, 15), edge_range=(10, 30)):
    """Small synthetic dataset with variable graph sizes."""
    import random
    random.seed(42)
    return [
        Data(
            x=torch.randn(random.randint(*node_range), 4),
            edge_index=torch.zeros(2, random.randint(*edge_range),
                                   dtype=torch.long),
        ) for _ in range(n)
    ]


def test_mps_freeze_at_returns_none_when_unavailable():
    """_mps_freeze_at returns None when MPS is not available."""
    with patch('torch.backends.mps.is_available', return_value=False):
        assert _mps_freeze_at(_make_dataset(), batch_size=4) is None


def test_mps_freeze_at_returns_none_without_api():
    """_mps_freeze_at returns None when freeze_graph_cache API is absent."""
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.mps', spec=[]):  # spec=[] means no attributes
        assert _mps_freeze_at(_make_dataset(), batch_size=4) is None


def test_mps_freeze_at_formula():
    """freeze_at >= 5 and matches the sqrt(eff_2d) formula."""
    dataset = _make_dataset(n=100)
    batch_size = 4

    node_counts = [int(d.num_nodes) for d in dataset]
    edge_counts = [int(d.num_edges) for d in dataset]
    std_n = statistics.stdev(node_counts)
    std_e = statistics.stdev(edge_counts)
    eff_2d = (math.sqrt(batch_size) * std_n * math.sqrt(batch_size) * std_e *
              2 * math.pi)
    expected = max(5, int(math.sqrt(eff_2d)))

    with patch('torch.backends.mps.is_available', return_value=True), \
         patch.object(torch.mps, 'freeze_graph_cache', create=True):
        result = _mps_freeze_at(dataset, batch_size)

    assert result == expected
    assert result >= 5


def test_mps_freeze_called_after_warmup():
    """DataLoader calls freeze_graph_cache() exactly once after warmup."""
    dataset = _make_dataset(n=40)

    mock_freeze = MagicMock()
    # Patch _mps_freeze_at to return 3 so freeze triggers within 10 batches
    # (the formula yields ~21 for this dataset, exceeding the 10-batch window)
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch.object(
             torch.mps, 'freeze_graph_cache', mock_freeze,
             create=True), \
         patch(
             'torch_geometric.loader.dataloader._mps_freeze_at',
             return_value=3):
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batches = list(loader)

    assert mock_freeze.call_count == 1
    assert len(batches) == 10  # 40 graphs / batch_size=4


def test_mps_freeze_noop_on_non_mps():
    """DataLoader iterates normally and never calls freeze on non-MPS."""
    dataset = _make_dataset(n=20)
    mock_freeze = MagicMock()

    with patch('torch.backends.mps.is_available', return_value=False), \
         patch.object(
             torch.mps, 'freeze_graph_cache', mock_freeze,
             create=True):
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        assert loader._mps_freeze_at is None
        batches = list(loader)

    assert mock_freeze.call_count == 0
    assert len(batches) == 5  # 20 graphs / batch_size=4

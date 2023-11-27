import multiprocessing
import sys
from collections import namedtuple

import pytest
import torch

from torch_geometric.data import Data, HeteroData, OnDiskDataset
from torch_geometric.loader import DataLoader
from torch_geometric.testing import (
    get_random_edge_index,
    get_random_tensor_frame,
    withCUDA,
    withPackage,
)

with_mp = sys.platform not in ['win32']
num_workers_list = [0, 2] if with_mp else [0]

if sys.platform == 'darwin':
    multiprocessing.set_start_method('spawn')


@withCUDA
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
        for batch in loader:
            pass
        print(f'Done! [{time.perf_counter() - t:.4f}s]')

    on_disk_dataset = in_memory_dataset.to_on_disk_dataset()
    loader = DataLoader(on_disk_dataset, **kwargs)

    print('On-Disk Dataset:')
    for _ in range(2):
        print(f'Start loading {len(loader)} mini-batches ... ', end='')
        t = time.perf_counter()
        for batch in loader:
            pass
        print(f'Done! [{time.perf_counter() - t:.4f}s]')

    on_disk_dataset.close()

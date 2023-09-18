import os.path as osp

import pytest
import torch

from torch_geometric.data.database import RocksDatabase, SQLiteDatabase
from torch_geometric.profile import benchmark
from torch_geometric.testing import withPackage


@withPackage('sqlite3')
@pytest.mark.parametrize('batch_size', [None, 1])
def test_sqlite_database(tmp_path, batch_size):
    path = osp.join(tmp_path, 'sqlite.db')
    db = SQLiteDatabase(path, name='test_table')
    assert str(db) == 'SQLiteDatabase(0)'
    assert len(db) == 0

    data = torch.randn(5)
    db.insert(0, data)
    assert len(db) == 1
    assert torch.equal(db.get(0), data)

    indices = torch.tensor([1, 2])
    data_list = torch.randn(2, 5)
    db.multi_insert(indices, data_list, batch_size=batch_size)
    assert len(db) == 3

    out_list = db.multi_get(indices, batch_size=batch_size)
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert torch.equal(out_list[0], data_list[0])
    assert torch.equal(out_list[1], data_list[1])

    db.close()


@withPackage('rocksdict')
@pytest.mark.parametrize('batch_size', [None, 1])
def test_rocks_database(tmp_path, batch_size):
    path = osp.join(tmp_path, 'rocks.db')
    db = RocksDatabase(path)
    assert str(db) == 'RocksDatabase()'
    with pytest.raises(NotImplementedError):
        len(db)

    data = torch.randn(5)
    db.insert(0, data)
    assert torch.equal(db.get(0), data)

    indices = torch.tensor([1, 2])
    data_list = torch.randn(2, 5)
    db.multi_insert(indices, data_list, batch_size=batch_size)

    out_list = db.multi_get(indices, batch_size=batch_size)
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert torch.equal(out_list[0], data_list[0])
    assert torch.equal(out_list[1], data_list[1])

    db.close()


@withPackage('sqlite3')
def test_database_syntactic_sugar(tmp_path):
    path = osp.join(tmp_path, 'sqlite.db')
    db = SQLiteDatabase(path, name='test_table')

    data = torch.randn(5, 16)
    db[0] = data[0]
    db[1:3] = data[1:3]
    db[torch.tensor([3, 4])] = data[torch.tensor([3, 4])]
    assert len(db) == 5

    assert torch.equal(db[0], data[0])
    assert torch.equal(torch.stack(db[:3], dim=0), data[:3])
    assert torch.equal(torch.stack(db[3:], dim=0), data[3:])
    assert torch.equal(torch.stack(db[1::2], dim=0), data[1::2])
    assert torch.equal(torch.stack(db[[4, 3]], dim=0), data[[4, 3]])
    assert torch.equal(
        torch.stack(db[torch.tensor([4, 3])], dim=0),
        data[torch.tensor([4, 3])],
    )
    assert torch.equal(
        torch.stack(db[torch.tensor([4, 4])], dim=0),
        data[torch.tensor([4, 4])],
    )


if __name__ == '__main__':
    import argparse
    import tempfile
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--numel', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    data = torch.randn(args.numel, 128)
    tmp_dir = tempfile.TemporaryDirectory()

    path = osp.join(tmp_dir.name, 'sqlite.db')
    sqlite_db = SQLiteDatabase(path, name='test_table')
    t = time.perf_counter()
    sqlite_db.multi_insert(range(args.numel), data, batch_size=100, log=True)
    print(f'Initialized SQLiteDB in {time.perf_counter() - t:.2f} seconds')

    path = osp.join(tmp_dir.name, 'rocks.db')
    rocks_db = RocksDatabase(path)
    t = time.perf_counter()
    rocks_db.multi_insert(range(args.numel), data, batch_size=100, log=True)
    print(f'Initialized RocksDB in {time.perf_counter() - t:.2f} seconds')

    def in_memory_get(data):
        index = torch.randint(0, args.numel, (args.batch_size, ))
        return data[index]

    def db_get(db):
        index = torch.randint(0, args.numel, (args.batch_size, ))
        return db[index]

    benchmark(
        funcs=[in_memory_get, db_get, db_get],
        func_names=['In-Memory', 'SQLite', 'RocksDB'],
        args=[(data, ), (sqlite_db, ), (rocks_db, )],
        num_steps=50,
        num_warmups=5,
    )

    tmp_dir.cleanup()

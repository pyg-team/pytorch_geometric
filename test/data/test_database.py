import math
import os.path as osp

import pytest
import torch

from torch_geometric import EdgeIndex, Index
from torch_geometric.data import Data, RocksDatabase, SQLiteDatabase
from torch_geometric.data.database import TensorInfo
from torch_geometric.profile import benchmark
from torch_geometric.testing import has_package, withPackage

AVAILABLE_DATABASES = []
if has_package('sqlite3'):
    AVAILABLE_DATABASES.append(SQLiteDatabase)
if has_package('rocksdict'):
    AVAILABLE_DATABASES.append(RocksDatabase)


@pytest.mark.parametrize('Database', AVAILABLE_DATABASES)
@pytest.mark.parametrize('batch_size', [None, 1])
def test_database_single_tensor(tmp_path, Database, batch_size):
    kwargs = dict(path=osp.join(tmp_path, 'storage.db'))
    if Database == SQLiteDatabase:
        kwargs['name'] = 'test_table'

    db = Database(**kwargs)
    assert db.schema == {0: object}

    try:
        assert len(db) == 0
        assert str(db) == f'{Database.__name__}(0)'
    except NotImplementedError:
        assert str(db) == f'{Database.__name__}()'

    data = torch.randn(5)
    db.insert(0, data)
    try:
        assert len(db) == 1
    except NotImplementedError:
        pass
    assert torch.equal(db.get(0), data)

    indices = torch.tensor([1, 2])
    data_list = torch.randn(2, 5)
    db.multi_insert(indices, data_list, batch_size=batch_size)
    try:
        assert len(db) == 3
    except NotImplementedError:
        pass
    out_list = db.multi_get(indices, batch_size=batch_size)
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert torch.equal(out_list[0], data_list[0])
    assert torch.equal(out_list[1], data_list[1])

    db.close()


@pytest.mark.parametrize('Database', AVAILABLE_DATABASES)
def test_database_schema(tmp_path, Database):
    kwargs = dict(name='test_table') if Database == SQLiteDatabase else {}

    path = osp.join(tmp_path, 'tuple_storage.db')
    schema = (int, float, str, dict(dtype=torch.float, size=(2, -1)), object)
    db = Database(path, schema=schema, **kwargs)
    assert db.schema == {
        0: int,
        1: float,
        2: str,
        3: TensorInfo(dtype=torch.float, size=(2, -1)),
        4: object,
    }

    data1 = (1, 0.1, 'a', torch.randn(2, 8), Data(x=torch.randn(8)))
    data2 = (2, float('inf'), 'b', torch.randn(2, 16), Data(x=torch.randn(8)))
    data3 = (3, float('NaN'), 'c', torch.randn(2, 32), Data(x=torch.randn(8)))
    db.insert(0, data1)
    db.multi_insert([1, 2], [data2, data3])

    out1 = db.get(0)
    out2, out3 = db.multi_get([1, 2])

    for out, data in zip([out1, out2, out3], [data1, data2, data3]):
        assert out[0] == data[0]
        if math.isnan(data[1]):
            assert math.isnan(out[1])
        else:
            assert out[1] == data[1]
        assert out[2] == data[2]
        assert torch.equal(out[3], data[3])
        assert isinstance(out[4], Data) and len(out[4]) == 1
        assert torch.equal(out[4].x, data[4].x)

    db.close()

    path = osp.join(tmp_path, 'dict_storage.db')
    schema = {
        'int': int,
        'float': float,
        'str': str,
        'tensor': dict(dtype=torch.float, size=(2, -1)),
        'data': object
    }
    db = Database(path, schema=schema, **kwargs)
    assert db.schema == {
        'int': int,
        'float': float,
        'str': str,
        'tensor': TensorInfo(dtype=torch.float, size=(2, -1)),
        'data': object,
    }

    data1 = {
        'int': 1,
        'float': 0.1,
        'str': 'a',
        'tensor': torch.randn(2, 8),
        'data': Data(x=torch.randn(1, 8)),
    }
    data2 = {
        'int': 2,
        'float': 0.2,
        'str': 'b',
        'tensor': torch.randn(2, 16),
        'data': Data(x=torch.randn(2, 8)),
    }
    data3 = {
        'int': 3,
        'float': 0.3,
        'str': 'c',
        'tensor': torch.randn(2, 32),
        'data': Data(x=torch.randn(3, 8)),
    }
    db.insert(0, data1)
    db.multi_insert([1, 2], [data2, data3])

    out1 = db.get(0)
    out2, out3 = db.multi_get([1, 2])

    for out, data in zip([out1, out2, out3], [data1, data2, data3]):
        assert out['int'] == data['int']
        assert out['float'] == data['float']
        assert out['str'] == data['str']
        assert torch.equal(out['tensor'], data['tensor'])
        assert isinstance(out['data'], Data) and len(out['data']) == 1
        assert torch.equal(out['data'].x, data['data'].x)

    db.close()


@pytest.mark.parametrize('Database', AVAILABLE_DATABASES)
def test_index(tmp_path, Database):
    kwargs = dict(name='test_table') if Database == SQLiteDatabase else {}

    path = osp.join(tmp_path, 'tuple_storage.db')
    schema = dict(dtype=torch.long, is_index=True)
    db = Database(path, schema=schema, **kwargs)
    assert db.schema == {
        0: TensorInfo(dtype=torch.long, is_index=True),
    }

    index1 = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    index2 = Index([0, 1, 1, 2, 2, 3], dim_size=None, is_sorted=True)
    index3 = Index([], dtype=torch.long)

    db.insert(0, index1)
    db.multi_insert([1, 2], [index2, index3])

    out1 = db.get(0)
    out2, out3 = db.multi_get([1, 2])

    for out, index in zip([out1, out2, out3], [index1, index2, index3]):
        assert index.equal(out)
        assert index.dtype == out.dtype
        assert index.dim_size == out.dim_size
        assert index.is_sorted == out.is_sorted

    db.close()


@pytest.mark.parametrize('Database', AVAILABLE_DATABASES)
def test_edge_index(tmp_path, Database):
    kwargs = dict(name='test_table') if Database == SQLiteDatabase else {}

    path = osp.join(tmp_path, 'tuple_storage.db')
    schema = dict(dtype=torch.long, is_edge_index=True)
    db = Database(path, schema=schema, **kwargs)
    assert db.schema == {
        0: TensorInfo(dtype=torch.long, size=(2, -1), is_edge_index=True),
    }

    adj1 = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
    )
    adj2 = EdgeIndex(
        [[1, 0, 2, 1, 3, 2], [0, 1, 1, 2, 2, 3]],
        sparse_size=(4, 4),
        sort_order='col',
    )
    adj3 = EdgeIndex([[], []], dtype=torch.long)

    db.insert(0, adj1)
    db.multi_insert([1, 2], [adj2, adj3])

    out1 = db.get(0)
    out2, out3 = db.multi_get([1, 2])

    for out, adj in zip([out1, out2, out3], [adj1, adj2, adj3]):
        assert adj.equal(out)
        assert adj.dtype == out.dtype
        assert adj.sparse_size() == out.sparse_size()
        assert adj.sort_order == out.sort_order
        assert adj.is_undirected == out.is_undirected

    db.close()


@withPackage('sqlite3')
def test_database_syntactic_sugar(tmp_path):
    path = osp.join(tmp_path, 'storage.db')
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

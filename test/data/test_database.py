import os.path as osp

import pytest
import torch

from torch_geometric.data.database import SQLiteDatabase
from torch_geometric.testing import withPackage


@withPackage('sqlite3')
@pytest.mark.parametrize('batch_size', [None, 1])
def test_sqlite_database(tmp_path, batch_size):
    path = osp.join(tmp_path, 'sqlite.db')
    db = SQLiteDatabase(path, name='test_table')
    assert str(db) == 'SQLiteDatabase()'
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

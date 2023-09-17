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

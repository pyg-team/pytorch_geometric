import abc
from collections import namedtuple
from dataclasses import dataclass
import io
import itertools
import json
from typing import Iterable, Generator, Optional, Any

import sqlite3
import torch

from torch_geometric.data.data import Data
from torch_geometric.typing import OptTensor


class GraphLabel:

    def __init__(self, id: str, *args, **kwargs):
        self.id = id
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class GraphRow:
    id: str
    data: Optional[dict[str, Optional[bytes]]]


def chunk(seq: Iterable, chunk_size: int) -> Generator[list, Any, None]:
    "chunk data into chunks of chunk_size"
    it = iter(seq)
    while True:
        batch = list(itertools.islice(it, chunk_size))
        if not batch:
            return
        yield batch


def namedtuple_factory(cursor, row):
    """util function to create a namedtuple Row foe db results"""
    fields = [column[0] for column in cursor.description]
    cls = namedtuple("Row", fields)
    return cls._make(row)


class Database(abc.ABC):

    def __init__(self, credentials, *args, **kwargs):
        self.connection = self._get_connection(credentials)

    @abc.abstractmethod
    def _initialize(self):
        """initialize the database in some way if needed"""
        raise NotImplementedError()

    @abc.abstractmethod
    def insert(self, labels: Iterable[GraphLabel], values: Iterable[Data], batch_size=10000) -> list[str]:
        """insert data into a database"""
        raise NotImplementedError()

    @abc.abstractmethod
    def serialize_data(self, data: Data) -> GraphRow:
        """serialize the data"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get(self, key: str) -> Data:
        """get data by key"""
        raise NotImplementedError()

    def multi_get(self, keys: list[str]) -> list[Data]:
        """get multiple keys"""
        return [self.get(key) for key in keys]


class SQLiteDatabase(abc.ABC):

    def __init__(self, credentials, table='pyg_database') -> None:
        self.table = table
        super().__init__(credentials)

    def _initialize(self):
        create = """CREATE TABLE ? (id TEXT, data, TEXT)"""
        self.cursor.execute(create, self.table)

    def insert(self, labels: Iterable[GraphLabel], values: Iterable[Data], batch_size=10000) -> list[GraphRow]:
        for chunk_data in chunk(zip(labels, values), batch_size):
            serialized = [self.serialize_data(label, value) for label, value in chunk_data]
            query = f"""
            INSERT INTO {self.table} (id, data) 
              VALUES (?, ?)"""
            self.cursor.executemany(query, [(row['id'], json.dumps(row['data'])) for row in serialized])

    def get(self, label: GraphLabel):
        query = f"""SELECT * FROM {self.table} where id = ?"""
        self.cursor.execute(query, (label.id))
        return self.cursor.fetchone()
    
    def multi_get(self, labels: Iterable[GraphLabel], batch_size=999):
        for chunk_data in chunk(labels, batch_size):
            query = f"SELECT * FROM {self.table} WHERE id IN ({','.join('?' * len(chunk_data))})"
            self.cursor.execute(query, (label.id for label in chunk_data))

    def serialize_data(self, label: GraphLabel, data: Data) -> GraphRow:
        row_dict = {k: self._serialize_tensor(v) if isinstance(v, OptTensor) else v for k, v in vars(data).items()}
        return GraphRow(
            **vars(label),
            **row_dict
        )

    @staticmethod
    def _serialize_tensor(t: OptTensor) -> bytes:
        """convert a tensor into bytes"""
        buff = io.BytesIO()
        torch.save(t, buff)
        return buff.getvalue()

    def _get_connection(self, credentials):
        """a method to get the db cursor to executor SQL"""
        con = sqlite3.connect(credentials)
        cursor = con.cursor()
        cursor.row_factory = namedtuple_factory
        return cursor

import abc
from collections import namedtuple
from dataclasses import dataclass
import io
import itertools
from typing import Iterable, Sequence, Optional, Any

import sqlite3
import torch

from torch_geometric.data.data import Data
from torch_geometric.typing import OptTensor

@dataclass
class GraphLabel:
    id: str
    type: str

@dataclass
class GraphRow:
    id: str
    type: str
    x: Optional[bytes]
    edge_index: Optional[bytes]
    edge_attr: Optional[bytes]
    y: Optional[bytes]
    pos: Optional[bytes]


def chunk(seq: Iterable, chunk_size: int) -> Generator[list, Any, None]:
    "chunk data into chunks of chunk_size"
    it = iter(seq)
    while True:
        batch = list(itertools.islice(it, chunk_size))
        if not batch:
            return
        yield batch

def namedtuple_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    cls = namedtuple("Row", fields)
    return cls._make(row)


class Database(abc.ABC):

    def __init__(self, connection):
        self.cursor = self._get_cursor(connection)

    @abc.abstractmethod
    def insert(elf, labels: Iterable[GraphLabel], values: Iterable[Data], batch_size=10000) -> list[str]:
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

    def __init__(self, connection, table='pyg_database') -> None:
        super().__init__(connection)

    def _initialize(self):
        create = """CREATE TABLE ? (id TEXT, type TEXT, x BLOB, edge_index BLOB, edge_attr BLOB, y BLOB, pos BLOB, meta TEXT)"""
        self.cursor.execute(create, self.table)


    def insert(self, labels: Iterable[GraphLabel], values: Iterable[Data], batch_size=10000) -> list[GraphRow]:
        for chunk_data in chunk(zip(labels, values), batch_size):
            serialized = [self.serialize_data(label, value) for value in chunk_data]
            query = f"""
            INSERT INTO {self.table} (id, type, x, edge_index, edge_attr, y, pos, meta) 
            VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?)"""
            self.cursor.executemany(query, [row.astuple() for row in serialized])

    def get(self, label: GraphLabel):
        query = f"""SELECT * FROM {self.table} where id = ? and type = ?"""
        self.cursor.execute(query, (label.id, label.type))
        return self.cursor.fetchone()
    
    def multi_get(self, labels: Iterable[GraphLabel], batch_size=999):
        for chunk_data in chunk(labels):
            query = f"SELECT * FROM {self.table} WHERE id IN ({','.join('?' * len(chunk_data))})"
            self.cursor.execute(query, (label.id for label in chunk_data))


    def serialize_data(self, label: GraphLabel, data: Data) -> GraphRow:
        return GraphRow(
            id=label.id,
            type=data.type,
            x=self._serialize_tensor(data.x),
            edge_index=self._serialize_tensor(data.edge_index),
            edge_attr=self._serialize_tensor(data.edge_attr),
            y=self._serialize_tensor(data.y),
            pos=self._serialize_tensor(data.pos)
        )

    @staticmethod
    def _serialize_tensor(t: OptTensor) -> bytes:
        buff = io.BytesIO()
        torch.save(t, buff)
        return buff.getvalue()
    
    def _get_cursor(self, connection):
        con = sqlite3.connect(connection)
        cursor = con.cursor()
        cursor.row_factory = namedtuple_factory




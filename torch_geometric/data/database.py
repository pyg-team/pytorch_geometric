import io
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Union

import torch
from torch import Tensor


class Database(ABC):
    r"""Base class for database."""
    def connect(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def insert(self, index: int, data: Any):
        raise NotImplementedError

    def multi_insert(
        self,
        indices: Union[Iterable[int], Tensor],
        data_list: Iterable[Any],
        batch_size: Optional[int] = None,
    ):
        if batch_size is None:
            batch_size = min(len(indices), len(data_list))

        for start in range(0, min(len(indices), len(data_list)), batch_size):
            self._multi_insert(
                indices[start:start + batch_size],
                data_list[start:start + batch_size],
            )

    def _multi_insert(
        self,
        indices: Union[Iterable[int], Tensor],
        data_list: Iterable[Any],
    ):
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        for index, data in zip(indices, data_list):
            self.insert(index, data)

    @abstractmethod
    def get(self, index: int) -> Any:
        raise NotImplementedError

    def multi_get(
        self,
        indices: Union[Iterable[int], Tensor],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        if batch_size is None:
            batch_size = len(indices)

        data_list: List[Any] = []
        for start in range(0, len(indices), batch_size):
            chunk_indices = indices[start:start + batch_size]
            data_list.extend(self._multi_get(chunk_indices))
        return data_list

    def _multi_get(self, indices: Union[Iterable[int], Tensor]) -> List[Any]:
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return [self.get(index) for index in indices]

    # Helper functions ########################################################

    @staticmethod
    def serialize(data: Any) -> bytes:
        r"""Serializes :obj:`data` into bytes."""
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def deserialize(data: bytes) -> Any:
        r"""Deserializes bytes into the original data."""
        return torch.load(io.BytesIO(data))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class SQLiteDatabase(Database):
    def __init__(self, path: str, name: str):
        super().__init__()

        import sqlite3

        self.path = path
        self.name = name

        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

        self.connect()

        query = (f'CREATE TABLE IF NOT EXISTS {self.name} (\n'
                 f'  id INTEGER PRIMARY KEY,\n'
                 f'  data BLOB NOT NULL\n'
                 f')')
        self.cursor.execute(query)

    def connect(self):
        import sqlite3
        self._connection = sqlite3.connect(self.path)
        self._cursor = self._connection.cursor()

    def close(self):
        self._connection.commit()
        self._connection.close()
        self._connection = None
        self._cursor = None

    @property
    def cursor(self) -> Any:
        if self._cursor is None:
            raise RuntimeError("No open database connection")
        return self._cursor

    def insert(self, index: int, data: Any):
        query = f'INSERT INTO {self.name} (id, data) VALUES (?, ?)'
        self.cursor.execute(query, (index, self.serialize(data)))

    def _multi_insert(
        self,
        indices: Union[Iterable[int], Tensor],
        data_list: Iterable[Any],
    ):
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        data_list = [self.serialize(data) for data in data_list]

        query = f'INSERT INTO {self.name} (id, data) VALUES (?, ?)'
        self.cursor.executemany(query, zip(indices, data_list))

    def get(self, index: int) -> Any:
        query = f'SELECT data FROM {self.name} WHERE id = ?'
        self.cursor.execute(query, (index, ))
        return self.deserialize(self.cursor.fetchone()[0])

    def multi_get(
        self,
        indices: Union[Iterable[int], Tensor],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        query = (f'SELECT data FROM {self.name} '
                 f'WHERE id IN ({", ".join("?" * len(indices))})')
        self.cursor.execute(query, indices)

        if batch_size is None:
            data_list = self.cursor.fetchall()
        else:
            data_list: List[Any] = []
            while True:
                chunk_list = self.cursor.fetchmany(size=batch_size)
                if len(chunk_list) == 0:
                    break
                data_list.extend(chunk_list)

        return [self.deserialize(data[0]) for data in data_list]

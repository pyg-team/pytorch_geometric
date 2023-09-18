import pickle
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Union
from uuid import uuid4

from torch import Tensor
from tqdm import tqdm


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
        indices: Union[Iterable[int], Tensor, slice, range],
        data_list: Iterable[Any],
        batch_size: Optional[int] = None,
        log: bool = False,
    ):
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)

        length = min(len(indices), len(data_list))
        batch_size = length if batch_size is None else batch_size

        if log and length > batch_size:
            desc = f'Insert {length} entries'
            offsets = tqdm(range(0, length, batch_size), desc=desc)
        else:
            offsets = range(0, length, batch_size)

        for start in offsets:
            self._multi_insert(
                indices[start:start + batch_size],
                data_list[start:start + batch_size],
            )

    def _multi_insert(
        self,
        indices: Union[Iterable[int], Tensor, range],
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
        indices: Union[Iterable[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)

        length = len(indices)
        batch_size = length if batch_size is None else batch_size

        data_list: List[Any] = []
        for start in range(0, length, batch_size):
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
        # Ensure that data is not a view of a larger tensor:
        if isinstance(data, Tensor):
            data = data.clone()

        return pickle.dumps(data)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        r"""Deserializes bytes into the original data."""
        return pickle.loads(data)

    def slice_to_range(self, indices: slice) -> range:
        start = 0 if indices.start is None else indices.start
        stop = len(self) if indices.stop is None else indices.stop
        step = 1 if indices.step is None else indices.step

        return range(start, stop, step)

    # Python built-ins ########################################################

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self,
        key: Union[int, Iterable[int], Tensor, slice, range],
    ) -> Union[Any, List[Any]]:

        if isinstance(key, int):
            return self.get(key)
        else:
            return self.multi_get(key)

    def __setitem__(
        self,
        key: Union[int, Iterable[int], Tensor, slice, range],
        value: Union[Any, Iterable[Any]],
    ):
        if isinstance(key, int):
            self.insert(key, value)
        else:
            self.multi_insert(key, value)

    def __repr__(self) -> str:
        try:
            return f'{self.__class__.__name__}({len(self)})'
        except NotImplementedError:
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
        if self._connection is not None:
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
        indices: Union[Iterable[int], Tensor, range],
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
        indices: Union[Iterable[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:

        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        # We first create a temporary ID table to then perform an INNER JOIN.
        # This avoids having a long IN clause and guarantees sorted outputs:
        join_table_name = f'{self.name}__join__{uuid4().hex}'
        query = (f'CREATE TABLE {join_table_name} (\n'
                 f'  id INTEGER,\n'
                 f'  row_id INTEGER\n'
                 f')')
        self.cursor.execute(query)

        query = f'INSERT INTO {join_table_name} (id, row_id) VALUES (?, ?)'
        self.cursor.executemany(query, zip(indices, range(len(indices))))

        query = f'SELECT * FROM {join_table_name}'
        self.cursor.execute(query)

        query = (f'SELECT {self.name}.data '
                 f'FROM {self.name} INNER JOIN {join_table_name} '
                 f'ON {self.name}.id = {join_table_name}.id '
                 f'ORDER BY {join_table_name}.row_id')
        self.cursor.execute(query)

        if batch_size is None:
            data_list = self.cursor.fetchall()
        else:
            data_list: List[Any] = []
            while True:
                chunk_list = self.cursor.fetchmany(size=batch_size)
                if len(chunk_list) == 0:
                    break
                data_list.extend(chunk_list)

        query = f'DROP TABLE {join_table_name}'
        self.cursor.execute(query)

        return [self.deserialize(data[0]) for data in data_list]

    def __len__(self) -> int:
        query = f'SELECT COUNT(*) FROM {self.name}'
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]


class RocksDatabase(Database):
    def __init__(self, path: str):
        super().__init__()

        import rocksdict

        self.path = path

        self._db: Optional[rocksdict.Rdict] = None

        self.connect()

    def connect(self):
        import rocksdict
        self._db = rocksdict.Rdict(
            self.path,
            options=rocksdict.Options(raw_mode=True),
        )

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def db(self) -> Any:
        if self._db is None:
            raise RuntimeError("No open database connection")
        return self._db

    @staticmethod
    def to_key(index: int) -> bytes:
        return index.to_bytes(8, byteorder='big', signed=True)

    def insert(self, index: int, data: Any):
        # Ensure that data is not a view of a larger tensor:
        if isinstance(data, Tensor):
            data = data.clone()

        self.db[self.to_key(index)] = self.serialize(data)

    def get(self, index: int) -> Any:
        return self.deserialize(self.db[self.to_key(index)])

    def _multi_get(self, indices: Union[Iterable[int], Tensor]) -> List[Any]:
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        indices = [self.to_key(index) for index in indices]
        data_list = self.db[indices]
        return [self.deserialize(data) for data in data_list]

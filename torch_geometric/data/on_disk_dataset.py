import os
from typing import Any, Callable, Iterable, List, Optional, Union

from torch import Tensor

from torch_geometric.data import Database, RocksDatabase, SQLiteDatabase
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import Schema
from torch_geometric.data.dataset import Dataset


class OnDiskDataset(Dataset):
    r"""Dataset base class for creating large graph datasets which do not
    easily fit into CPU memory at once by leveraging a :class:`Database`
    backend for on-disk storage and access of data objects.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        backend (str): The :class:`Database` backend to use
            (one of :obj:`"sqlite"` or :obj:`"rocksdb"`).
            (default: :obj:`"sqlite"`)
        schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
            the input data.
            Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
            dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
            tensor data) as input, and can be nested as a tuple or dictionary.
            Specifying the schema will improve efficiency, since by default the
            database will use python pickling for serializing and
            deserializing. If specified to anything different than
            :obj:`object`, implementations of :class:`OnDiskDataset` need to
            override :meth:`serialize` and :meth:`deserialize` methods.
            (default: :obj:`object`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
    """
    BACKENDS = {
        'sqlite': SQLiteDatabase,
        'rocksdb': RocksDatabase,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        backend: str = 'sqlite',
        schema: Schema = object,
        log: bool = True,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(f"Database backend must be one of "
                             f"{set(self.BACKENDS.keys())} "
                             f"(got '{backend}')")

        self.backend = backend
        self.schema = schema

        self._db: Optional[Database] = None
        self._numel: Optional[int] = None

        super().__init__(root, transform, pre_filter=pre_filter, log=log)

    @property
    def processed_file_names(self) -> str:
        return f'{self.backend}.db'

    @property
    def db(self) -> Database:
        r"""Returns the underlying :class:`Database`."""
        if self._db is not None:
            return self._db

        kwargs = {}
        cls = self.BACKENDS[self.backend]
        if issubclass(cls, SQLiteDatabase):
            kwargs['name'] = self.__class__.__name__

        os.makedirs(self.processed_dir, exist_ok=True)
        path = self.processed_paths[0]
        self._db = cls(path=path, schema=self.schema, **kwargs)
        self._numel = len(self._db)
        return self._db

    def close(self):
        r"""Closes the connection to the underlying database."""
        if self._db is not None:
            self._db.close()

    def serialize(self, data: BaseData) -> Any:
        r"""Serializes the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object into the expected DB
        schema.
        """
        if self.schema == object:
            return data
        raise NotImplementedError(f"`{self.__class__.__name__}.serialize()` "
                                  f"needs to be overridden in case a "
                                  f"non-default schema was passed")

    def deserialize(self, data: Any) -> BaseData:
        r"""Deserializes the DB entry into a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object.
        """
        if self.schema == object:
            return data
        raise NotImplementedError(f"`{self.__class__.__name__}.deserialize()` "
                                  f"needs to be overridden in case a "
                                  f"non-default schema was passed")

    def append(self, data: BaseData):
        r"""Appends the data object to the dataset."""
        index = len(self)
        self.db.insert(index, self.serialize(data))
        self._numel += 1

    def extend(
        self,
        data_list: List[BaseData],
        batch_size: Optional[int] = None,
    ):
        r"""Extends the dataset by a list of data objects."""
        start = len(self)
        end = start + len(data_list)
        data_list = [self.serialize(data) for data in data_list]
        self.db.multi_insert(range(start, end), data_list, batch_size)
        self._numel += (end - start)

    def get(self, idx: int) -> BaseData:
        r"""Gets the data object at index :obj:`idx`."""
        return self.deserialize(self.db.get(idx))

    def multi_get(
        self,
        indices: Union[Iterable[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[BaseData]:
        r"""Gets a list of data objects from the specified indices."""
        if len(indices) == 1:
            data_list = [self.db.get(indices[0])]
        else:
            data_list = self.db.multi_get(indices, batch_size)

        return [self.deserialize(data) for data in data_list]

    def len(self) -> int:
        if self._numel is None:
            self._numel = len(self.db)
        return self._numel

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'

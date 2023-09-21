from typing import Iterable, List, Optional, Callable, Union

from torch import Tensor

from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import SQLiteDatabase, RocksDatabase
from torch_geometric.data.dataset import Dataset


class OnDiskDatasetError(Exception):
    pass


class OnDiskDataset(Dataset):

    BACKENDS = {
            'rocksdb': RocksDatabase,
            'sqlite': SQLiteDatabase
    }

    def __init__(
        self,
        backend: str = 'rocksdb',
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        *args,
        **kwargs
    ):
    
        self.db = self._init_db(backend, *args, **kwargs)

        super.__init__(root, transform, pre_transform, pre_filter, log)
        self._data = None
        self.slices = None
        self._data_list: Optional[List[BaseData]] = None

    def _init_db(self, backend, *args, **kwargs):
        if backend not in self.BACKENDS:
            raise OnDiskDatasetError(f'backend must be one of: {list(self.BACKENDS.keys())}')
        backend_cls = self.BACKENDS[backend]
        return backend_cls(*args, **kwargs)

    def append_data(self, data_list: List[BaseData], batch_size=None, log=True):
        if len(data_list) == 1:
            self.db.insert(data_list[0])
        if isinstance(data_list, list):
            data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
            )
            self.db.multi_insert(slices, data, batch_size, log)

    def get(self, idx):
        return self.db.get(idx)

    def multi_get(self, indices: Union[Iterable[int], Tensor, slice, range], batch_size):
        if len(indices) == 1:
            return self.get(indices[0])
        else:
            return self.db.multi_get(indices, batch_size)
        
    def len(self) -> int: 
        return len(self.db)
    
    
    


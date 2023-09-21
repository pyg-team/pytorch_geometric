from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Callable, Union

from torch import Tensor

from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import SQLiteDatabase, Schema
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.in_memory_dataset import nested_iter



class OnDiskDataset(Dataset, ABC):
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        super.__init__(root, transform, pre_transform, pre_filter, log)
        self._data = None
        self.slices = None
        self._data_list: Optional[List[BaseData]] = None

    @abstractmethod
    def init_db(self):
        raise NotImplementedError
    
    @abstractmethod
    def append_data(self):
        raise NotImplementedError
    
    @abstractmethod
    def get(self, idx):
        raise NotImplementedError
    

class SQLiteDataset(OnDiskDataset):

    def __init__(self,
                 path: str,
                 name: str,
                 schema: Schema,
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None, 
                 pre_filter: Optional[Callable] = None, 
                 log: bool = True
    ):
        
        super().__init__(path, transform, pre_transform, pre_filter, log)
        self.db = self.init_db(path, name, schema)

    def init_db(self, path, name, schema):
        return SQLiteDatabase(path, name, schema)
    
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


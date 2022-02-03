from typing import Union, List, Optional

from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData

class _SingleProcessTemporalDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessTemporalDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset[index]
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


class TemporalDataLoader(DataLoader):
    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessTemporalDataLoaderIter(self)
        else:
            return super().__iter__()

from typing import Callable, Any

from torch.utils.data import DataLoader, _BaseDataLoaderIter


class DataLoaderIterator(object):
    def __init__(self, iterator: _BaseDataLoaderIter, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))


class BaseDataLoader(DataLoader):
    def _get_iterator(self) -> DataLoaderIterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

from typing import Callable, Any, Iterator

from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter


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
    r"""Extends the :class:`torch.utils.data.DataLoader` by integrating a
    custom :meth:`self.transform_fn` function to allow transformation of a
    returned mini-batch directly inside the main process.
    """
    def _get_iterator(self) -> Iterator:
        iterator = super()._get_iterator()
        if hasattr(self, 'transform_fn'):
            iterator = DataLoaderIterator(iterator, self.transform_fn)
        return iterator

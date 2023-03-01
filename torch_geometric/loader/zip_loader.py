from typing import Any, Iterator, List, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkLoader, NodeLoader
from torch_geometric.loader.base import DataLoaderIterator


class ZipLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        loaders: List[Union[NodeLoader, LinkLoader]],
        filter_per_worker: bool = False,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        for loader in loaders:
            if not callable(getattr(loader, 'collate_fn', None)):
                raise ValueError("'{loader.__class__.__name__}' does not have "
                                 "a 'collate_fn' method")
            if not callable(getattr(loader, 'filter_fn', None)):
                raise ValueError("'{loader.__class__.__name__}' does not have "
                                 "a 'filter_fn' method")
            loader = loader.filter_per_worker = filter_per_worker

        iterator = range(min([len(loader.dataset) for loader in loaders]))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

        self.loaders = loaders

    def collate_fn(self, index: List[int]) -> Tuple[Any, ...]:
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return tuple(loader.collate_fn(index) for loader in self.loaders)

    def filter_fn(
        self,
        outs: Tuple[Any, ...],
    ) -> Tuple[Union[Data, HeteroData], ...]:
        loaders = self.loaders
        return tuple(loader.filter_fn(v) for loader, v in zip(loaders, outs))

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(loaders={self.loaders})'

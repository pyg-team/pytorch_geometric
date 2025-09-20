from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkLoader, NodeLoader
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import infer_filter_per_worker


class ZipLoader(torch.utils.data.DataLoader):
    r"""A loader that returns a tuple of data objects by sampling from multiple
    :class:`NodeLoader` or :class:`LinkLoader` instances.

    Args:
        loaders (List[NodeLoader] or List[LinkLoader]): The loader instances.
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returned data in each worker's subprocess.
            If set to :obj:`False`, will filter the returned data in the main
            process.
            If set to :obj:`None`, will automatically infer the decision based
            on whether data partially lives on the GPU
            (:obj:`filter_per_worker=True`) or entirely on the CPU
            (:obj:`filter_per_worker=False`).
            There exists different trade-offs for setting this option.
            Specifically, setting this option to :obj:`True` for in-memory
            datasets will move all features to shared memory, which may result
            in too many open file handles. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        loaders: Union[List[NodeLoader], List[LinkLoader]],
        filter_per_worker: Optional[bool] = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(loaders[0].data)

        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        for loader in loaders:
            if not callable(getattr(loader, 'collate_fn', None)):
                raise ValueError("'{loader.__class__.__name__}' does not have "
                                 "a 'collate_fn' method")
            if not callable(getattr(loader, 'filter_fn', None)):
                raise ValueError("'{loader.__class__.__name__}' does not have "
                                 "a 'filter_fn' method")
            loader.filter_per_worker = filter_per_worker

        iterator = range(min([len(loader.dataset) for loader in loaders]))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

        self.loaders = loaders
        self.filter_per_worker = filter_per_worker

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Tuple[Data, ...], Tuple[HeteroData, ...]]:
        r"""Samples subgraphs from a batch of input IDs."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

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

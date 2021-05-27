from typing import Optional, Callable, List, Union, Tuple, Dict

import copy
from itertools import repeat
from inspect import signature
from collections.abc import Mapping

import torch
import numpy as np
from torch import Tensor

from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into CPU memory.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = None
        self.slices = None
        self._data_list: Optional[List[BaseData]] = None

    @property
    def num_classes(self) -> int:
        r"""The number of classes in the dataset."""
        y = self.data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(self.data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return self.data.y.size(-1)

    def len(self) -> int:
        if self.slices is None:
            return 1

        def _recurse(slices):
            for value in slices.values():
                if not isinstance(value, Mapping):
                    return len(value) - 1
                else:
                    return _recurse(value)

        return _recurse(self.slices)

    def get(self, idx: int) -> BaseData:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        num_args = len(signature(self.data.__cat_dim__).parameters)

        def _recursive_slice_(inputs, slices):
            for key, slice_info in slices.items():
                if isinstance(slice_info, Mapping):
                    _recursive_slice_(inputs[key], slice_info)
                else:
                    value = inputs[key]
                    start, end = int(slice_info[idx]), int(slice_info[idx + 1])
                    if isinstance(value, (Tensor, np.ndarray)):
                        slice_obj = list(repeat(slice(None), value.ndim))
                        # TODO: Inputs might not be a storage
                        args = (key, value, inputs)[:num_args]
                        dim = self.data.__cat_dim__(*args)
                        dim = 0 if dim is None else dim
                        slice_obj[dim] = slice(start, end)
                    elif start + 1 == end:
                        slice_obj = slice_info[start]
                    else:
                        slice_obj = slice(start, end)
                    inputs[key] = value[slice_obj]

        data = copy.copy(self.data)
        _recursive_slice_(data, self.slices)
        self._data_list[idx] = copy.copy(data)

        return data

    @staticmethod
    def collate(
        data_list: List[BaseData]
    ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data = copy.copy(data_list[0])
        slices = {}

        num_args = len(signature(data.__cat_dim__).parameters)

        def _recursive_concat_(outputs, inputs, slices):
            for key, value in outputs.items():
                if isinstance(value, Mapping):
                    slices[key] = {}
                    inputs = [v[key] for v in inputs]
                    _recursive_concat_(value, inputs, slices[key])
                else:
                    if isinstance(value, Tensor):  # TODO NUMPY
                        args = (key, value, outputs)[:num_args]
                        dim = data.__cat_dim__(*args)
                        dim = 0 if dim is None else dim
                        outputs[key] = [v[key] for v in inputs]
                        if value.ndim > 0:
                            slices[key] = [0]
                            slices[key] += [v[key].shape[dim] for v in inputs]
                            outputs[key] = torch.cat(outputs[key], dim)
                        else:
                            slices[key] = range(len(inputs) + 1)
                            outputs[key] = torch.stack(outputs[key])
                    else:
                        slices[key] = range(len(inputs) + 1)
                        outputs[key] = [v[key] for v in inputs]

                    slices[key] = torch.tensor(slices[key])

        # stores_2d: [num_stores, len(data_list)]
        # TODO: _stores do not have to be stored in the same order
        stores_2D = list(zip(*[elem._stores for elem in data_list]))
        for store, stores in zip(data._stores, stores_2D):
            if hasattr(store, '_key') and store._key is not None:
                slices[store._key] = {}
                _recursive_concat_(store, stores, slices[store._key])
            else:
                _recursive_concat_(store, stores, slices)

        return data, slices

    def copy(self, idx: Optional[IndexType] = None):
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = self.collate(data_list)
        return dataset

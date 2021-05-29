from typing import Optional, Callable, List, Union, Tuple, Dict, Iterable

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
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get(self, idx: int) -> BaseData:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = copy.copy(self.data)
        num_args = len(signature(data.__cat_dim__).parameters)

        def _slice(output: Mapping, slices, store):
            # * `output` denotes a mapping holding (child) elements of a copy
            #   from `self.data` which we want to slice to only contain values
            #   of the data element at position `idx`.
            # * `slices` is a dictionary that holds information about how we
            #   can reconstruct individual elements from a collated
            #   representation, e.g.: `slices = { 'x': [0, 6, 10] }`
            # * `store` denotes the parent store to which output elements
            #   belong to
            for key, value in output.items():
                if isinstance(value, Mapping):
                    _slice(value, slices[key], store)
                else:
                    start = int(slices[key][idx])
                    end = int(slices[key][idx + 1])

                    if isinstance(value, (Tensor, np.ndarray)):
                        slice_obj = list(repeat(slice(None), value.ndim))
                        args = (key, value, store)[:num_args]
                        dim = self.data.__cat_dim__(*args)
                        if dim is not None:
                            slice_obj[dim] = slice(start, end)
                        else:
                            slice_obj[0] = start
                    elif start + 1 == end:
                        slice_obj = start
                    else:
                        slice_obj = slice(start, end)

                    output[key] = value[slice_obj]

        for store in data._stores:
            if hasattr(store, '_key') and store._key is not None:
                _slice(store, self.slices[store._key], store)
            else:
                _slice(store, self.slices, store)

        self._data_list[idx] = copy.copy(data)

        return data

    @staticmethod
    def collate(
        data_list: List[BaseData]
    ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:
        r"""Collates a Python list of data objects to the internal storage
        format of :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        slices = {}
        data = copy.copy(data_list[0])
        num_args = len(signature(data.__cat_dim__).parameters)

        def _cat(inputs: List[Mapping], output: Mapping, slices, store):
            # * `inputs` is a list of dictionaries holding (child) elements of
            #   `data_list`
            # * `output` is the corresponding output mapping that stores the
            #   corresponding elements of `inputs` in a collated representation
            # * `slices` is a dictionary that holds information about how we
            #   can reconstruct individual elements from a collated
            #   representation, e.g.: `slices = { 'x': [0, 6, 10] }`
            # * `store` denotes the parent store to which inputs and output
            #   elements belong to
            for key, value in output.items():
                if isinstance(value, Mapping):
                    slices[key] = {}
                    _cat([v[key] for v in inputs], value, slices[key], store)
                else:
                    if isinstance(value, (Tensor, np.ndarray)):
                        args = (key, value, store)[:num_args]
                        dim = data.__cat_dim__(*args)
                        output[key] = [v[key] for v in inputs]
                        if value.ndim > 0 and dim is not None:
                            slices[key] = [0]
                            for v in output[key]:
                                size = v.shape[dim]
                                slices[key].append(slices[key][-1] + size)
                            if isinstance(value, Tensor):
                                output[key] = torch.cat(output[key], dim)
                            else:
                                output[key] = np.concatenate(output[key], dim)
                        else:
                            slices[key] = range(len(inputs) + 1)
                            if isinstance(value, Tensor):
                                output[key] = torch.stack(output[key], 0)
                            else:
                                output[key] = np.stack(output[key], 0)
                    else:
                        slices[key] = range(len(inputs) + 1)
                        output[key] = [v[key] for v in inputs]

                    slices[key] = torch.tensor(slices[key])

        # We group all storage objects of every data object in the
        # `data_list` by key, i.e. `stores_dict: { store_key: [store, ...] }`.
        stores_dict = {}
        for elem in data_list:
            for key, store in elem._store_dict.items():
                stores_dict[key] = stores_dict.get(key, []) + [store]

        # After which we concatenate all elements of all attributes in
        # `data_list` together by recursively iterating over each store.
        for key, store in data._store_dict.items():
            if hasattr(store, '_key') and store._key is not None:
                slices[store._key] = {}
                _cat(stores_dict[key], store, slices[store._key], store)
            else:
                _cat(stores_dict[key], store, slices, store)

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


def nested_iter(mapping: Mapping) -> Iterable:
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            for inner_key, inner_value, parent in nested_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value

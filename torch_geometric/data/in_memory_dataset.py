import copy
import warnings
from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate


class InMemoryDataset(Dataset, ABC):
    r"""Dataset base class for creating graph datasets which easily fit
    into CPU memory.
    Inherits from :class:`torch_geometric.data.Dataset`.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
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
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self._data = None
        self.slices = None
        self._data_list: Optional[List[Data]] = None

    @property
    def num_classes(self) -> int:
        if self.transform is None:
            return self._infer_num_classes(self._data.y)
        return super().num_classes

    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data

    @staticmethod
    def collate(
            data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if idx is None:
            data_list = [self.get(i) for i in self.indices()]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = None
        dataset.data, dataset.slices = self.collate(data_list)
        return dataset

    @property
    def data(self) -> Any:
        msg1 = ("It is not recommended to directly access the internal "
                "storage format `data` of an 'InMemoryDataset'.")
        msg2 = ("The given 'InMemoryDataset' only references a subset of "
                "examples of the full dataset, but 'data' will contain "
                "information of the full dataset.")
        msg3 = ("The data of the dataset is already cached, so any "
                "modifications to `data` will not be reflected when accessing "
                "its elements. Clearing the cache now by removing all "
                "elements in `dataset._data_list`.")
        msg4 = ("If you are absolutely certain what you are doing, access the "
                "internal storage via `InMemoryDataset._data` instead to "
                "suppress this warning. Alternatively, you can access stacked "
                "individual attributes of every graph via "
                "`dataset.{attr_name}`.")

        msg = msg1
        if self._indices is not None:
            msg += f' {msg2}'
        if self._data_list is not None:
            msg += f' {msg3}'
            self._data_list = None
        msg += f' {msg4}'

        warnings.warn(msg)

        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._data_list = None

    def __getattr__(self, key: str) -> Any:
        data = self.__dict__.get('_data')
        if isinstance(data, Data) and key in data:
            if self._indices is None and data.__inc__(key, data[key]) == 0:
                return data[key]
            else:
                data_list = [self.get(i) for i in self.indices()]
                return Batch.from_data_list(data_list)[key]

        raise AttributeError(f"'{self.__class__.__name__}' object has no "
                             f"attribute '{key}'")


def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):
        for key, value in node.items():
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node

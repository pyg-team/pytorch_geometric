import copy
import os.path as osp
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import Tensor
from tqdm import tqdm

import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate
from torch_geometric.io import fs


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which easily fit
    into CPU memory.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log,
                         force_reload)

        self._data: Optional[BaseData] = None
        self.slices: Optional[Dict[str, Tensor]] = None
        self._data_list: Optional[MutableSequence[Optional[BaseData]]] = None

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

    def get(self, idx: int) -> BaseData:
        # TODO (matthias) Avoid unnecessary copy here.
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

    @classmethod
    def save(cls, data_list: Sequence[BaseData], path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        fs.torch_save((data.to_dict(), slices, data.__class__), path)

    def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    @staticmethod
    def collate(
        data_list: Sequence[BaseData],
    ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:
        r"""Collates a list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects to the internal
        storage format of :class:`~torch_geometric.data.InMemoryDataset`.
        """
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

    def to_on_disk_dataset(
        self,
        root: Optional[str] = None,
        backend: str = 'sqlite',
        log: bool = True,
    ) -> 'torch_geometric.data.OnDiskDataset':
        r"""Converts the :class:`InMemoryDataset` to a :class:`OnDiskDataset`
        variant. Useful for distributed training and hardware instances with
        limited amount of shared memory.

        root (str, optional): Root directory where the dataset should be saved.
            If set to :obj:`None`, will save the dataset in
            :obj:`root/on_disk`.
            Note that it is important to specify :obj:`root` to account for
            different dataset splits. (optional: :obj:`None`)
        backend (str): The :class:`Database` backend to use.
            (default: :obj:`"sqlite"`)
        log (bool, optional): Whether to print any console output while
            processing the dataset. (default: :obj:`True`)
        """
        if root is None and (self.root is None or not osp.exists(self.root)):
            raise ValueError(f"The root directory of "
                             f"'{self.__class__.__name__}' is not specified. "
                             f"Please pass in 'root' when creating on-disk "
                             f"datasets from it.")

        root = root or osp.join(self.root, 'on_disk')

        in_memory_dataset = self
        ref_data = in_memory_dataset.get(0)
        if not isinstance(ref_data, Data):
            raise NotImplementedError(
                f"`{self.__class__.__name__}.to_on_disk_dataset()` is "
                f"currently only supported on homogeneous graphs")

        # Parse the schema ====================================================

        schema: Dict[str, Any] = {}
        for key, value in ref_data.to_dict().items():
            if isinstance(value, (int, float, str)):
                schema[key] = value.__class__
            elif isinstance(value, Tensor) and value.dim() == 0:
                schema[key] = dict(dtype=value.dtype, size=(-1, ))
            elif isinstance(value, Tensor):
                size = list(value.size())
                size[ref_data.__cat_dim__(key, value)] = -1
                schema[key] = dict(dtype=value.dtype, size=tuple(size))
            else:
                schema[key] = object

        # Create the on-disk dataset ==========================================

        class OnDiskDataset(torch_geometric.data.OnDiskDataset):
            def __init__(
                self,
                root: str,
                transform: Optional[Callable] = None,
            ):
                super().__init__(
                    root=root,
                    transform=transform,
                    backend=backend,
                    schema=schema,
                )

            def process(self):
                _iter = [
                    in_memory_dataset.get(i)
                    for i in in_memory_dataset.indices()
                ]
                if log:  # pragma: no cover
                    _iter = tqdm(_iter, desc='Converting to OnDiskDataset')

                data_list: List[Data] = []
                for i, data in enumerate(_iter):
                    data_list.append(data)
                    if i + 1 == len(in_memory_dataset) or (i + 1) % 1000 == 0:
                        self.extend(data_list)
                        data_list = []

            def serialize(self, data: Data) -> Dict[str, Any]:
                return data.to_dict()

            def deserialize(self, data: Dict[str, Any]) -> Data:
                return Data.from_dict(data)

            def __repr__(self) -> str:
                arg_repr = str(len(self)) if len(self) > 1 else ''
                return (f'OnDisk{in_memory_dataset.__class__.__name__}('
                        f'{arg_repr})')

        return OnDiskDataset(root, transform=in_memory_dataset.transform)

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

        warnings.warn(msg, stacklevel=2)

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

    def to(self, device: Union[int, str]) -> 'InMemoryDataset':
        r"""Performs device conversion of the whole dataset."""
        if self._indices is not None:
            raise ValueError("The given 'InMemoryDataset' only references a "
                             "subset of examples of the full dataset")
        if self._data_list is not None:
            raise ValueError("The data of the dataset is already cached")
        self._data.to(device)
        return self

    def cpu(self, *args: str) -> 'InMemoryDataset':
        r"""Moves the dataset to CPU memory."""
        return self.to(torch.device('cpu'))

    def cuda(
        self,
        device: Optional[Union[int, str]] = None,
    ) -> 'InMemoryDataset':
        r"""Moves the dataset toto CUDA memory."""
        if isinstance(device, int):
            device = f'cuda:{int}'
        elif device is None:
            device = 'cuda'
        return self.to(device)


def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):
        for value in node.values():
            yield from nested_iter(value)
    elif isinstance(node, Sequence):
        yield from enumerate(node)
    else:
        yield None, node

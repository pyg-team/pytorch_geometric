from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch.utils.data
import torch_sparse
from beartype.typing import Any, List, Optional, Sequence, Tuple, Union
from torch.utils.data.dataloader import default_collate

import torch_geometric
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.data.storage import BaseStorage
from torch_geometric.typing import SparseTensor, TensorFrame, torch_frame
from torch_geometric.utils import is_sparse, is_torch_sparse_tensor
from torch_geometric.utils.sparse import cat

FLOAT_PADDING_VALUE = 1e-5
NON_FLOAT_PADDING_VALUE = -1


def _dense_pad_tensor(
    values,
    float_padding_value: float = FLOAT_PADDING_VALUE,
    non_float_padding_value: int = NON_FLOAT_PADDING_VALUE,
) -> Tuple[Any, Any]:
    """Densely pads a list of `torch.Tensor` along a new `batch_dim=0`.

    Args:
        values (list): A list of values to be padded.
        float_padding_value (float, optional): The padding value for `float` dtype tensors.
        non_float_padding_value (int, optional): The padding value for non-`float` (e.g., `int`) dtype tensors.

    Returns:
        A tuple containing the padded value and an optional mask.
    """
    mask = None
    cat_dim = None
    elem = values[0]
    dtype = elem.dtype
    padding_value = (float_padding_value if torch.is_floating_point(elem) else
                     non_float_padding_value)
    if elem.dim() == 0:
        values = [value.unsqueeze(0) for value in values]
    else:
        if dtype != torch.float:
            if dtype in [torch.uint8, torch.bool]:
                # NOTE: We cannot use `torch.nn.utils.rnn.pad_sequence` directly with unsigned integer/boolean tensors.
                values = [value.long() for value in values]
            if elem.dim() == 2 and elem.shape[0] == 2:
                # Concatenate `edge_index` tensors.
                # NOTE: Here, we assume dimension `0` denotes the source and target node indices and dimension `1` denotes the number of edges.
                values = [
                    value.permute(1, 0).unsqueeze(0)
                    for value in torch.nn.utils.rnn.pad_sequence(
                        [val.permute(1, 0) for val in values],
                        batch_first=True,
                        padding_value=padding_value,
                    )
                ]
            else:
                values = [
                    value.unsqueeze(0)
                    for value in torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True, padding_value=padding_value)
                ]
            if dtype in [torch.uint8, torch.bool]:
                # NOTE: We cannot use `torch.nn.utils.rnn.pad_sequence` directly with unsigned integer/boolean tensors.
                mask = torch.cat([(value != padding_value)
                                  for value in values], dim=cat_dim or 0)
                for value in values:
                    value[value == padding_value] = 0
                values = [value.to(dtype) for value in values]
        else:
            values = [
                value.unsqueeze(0)
                for value in torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=padding_value)
            ]

    return values, mask


def _dense_padded_collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    float_padding_value: float = FLOAT_PADDING_VALUE,
    non_float_padding_value: int = NON_FLOAT_PADDING_VALUE,
) -> Tuple[Any, Any]:
    """Collates a list of values into a single tensor or list of tensors.

    Args:
        key (str): The key of the attribute to be collated.
        values (list): A list of values to be collated.
        data_list (list): A list of data objects.
        stores (list): A list of storage objects.
        float_padding_value (float, optional): The padding value for `float` dtype tensors.
        non_float_padding_value (int, optional): The padding value for non-`float` (e.g., `int`) dtype tensors.

    Returns:
        A tuple containing the collated value and an optional mask.
    """
    cat_dim = None
    elem = values[0]

    if isinstance(elem, torch.Tensor) and not is_sparse(elem):
        # Concatenate a list of `torch.Tensor` along a new `batch_dim=0`.
        padding_value = (float_padding_value if torch.is_floating_point(elem)
                         else non_float_padding_value)
        values, mask = _dense_pad_tensor(
            values,
            float_padding_value=float_padding_value,
            non_float_padding_value=non_float_padding_value,
        )

        if getattr(elem, "is_nested", False):
            raise NotImplementedError(
                "Dense padding collation for nested tensors is not supported (tested) yet."
            )
            tensors = []
            for nested_tensor in values:
                tensors.extend(nested_tensor.unbind())
            value = torch.nested.nested_tensor(tensors)
            mask = torch.nested.map(lambda tensor: (tensor != padding_value),
                                    value)

            return value, mask

        out = None
        if torch.utils.data.get_worker_info() is not None:
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            if torch_geometric.typing.WITH_PT20:
                storage = elem.untyped_storage()._new_shared(
                    numel * elem.element_size(), device=elem.device)
            elif torch_geometric.typing.WITH_PT112:
                storage = elem.storage()._new_shared(numel, device=elem.device)
            else:
                storage = elem.storage()._new_shared(numel)
            shape = [len(data_list)] + list(values[np.argmax(
                [value.numel() for value in values])].shape[1:])
            out = elem.new(storage).resize_(shape)

        value = torch.cat(values, dim=cat_dim or 0, out=out)
        mask = mask if mask is not None else (value != padding_value)

        return value, mask

    elif isinstance(elem, TensorFrame):
        raise NotImplementedError(
            "Dense padding collation for TensorFrames is not supported (tested) yet."
        )
        values, mask = _dense_pad_tensor(
            values, non_float_padding_value=non_float_padding_value)
        value = torch_frame.cat(values, along="row")
        return value, mask

    elif is_sparse(elem):
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        raise NotImplementedError(
            "Dense padding collation for SparseTensors is not supported (tested) yet."
        )
        values, mask = _dense_pad_tensor(
            values, non_float_padding_value=non_float_padding_value)
        if is_torch_sparse_tensor(elem):
            value = cat(values, dim=cat_dim)
        else:
            value = torch_sparse.cat(values, dim=cat_dim)
        return value, mask

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        return value, None

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, mask_dict = {}, {}
        for key in elem.keys():
            value_dict[key], mask_dict[key] = _dense_padded_collate(
                key, [v[key] for v in values], data_list, stores)
        return value_dict, mask_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str)
          and len(elem) > 0 and isinstance(elem[0],
                                           (torch.Tensor, SparseTensor))):
        # Recursively collate elements of lists.
        value_list, mask_list = [], []
        for i in range(len(elem)):
            value, mask = _dense_padded_collate(key, [v[i] for v in values],
                                                data_list, stores)
            value_list.append(value)
            mask_list.append(mask)
        return value_list, mask_list

    else:
        # Other-wise, just return the list of values as it is.
        return values, None


def dense_padded_collate(
    cls,
    data_list: List[BaseData],
    follow_batch: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> Tuple[BaseData, Mapping]:
    """Collates a list of `data` objects into a single dense object of type `cls`. `collate` can
    handle both homogeneous and heterogeneous data objects by individually collating all their
    stores. In addition, `collate` can handle nested data structures such as dictionaries and
    lists.

    Args:
        cls (type): The class into which to collate.
        data_list (list): A list of data objects.
        follow_batch (list, optional): Creates assignment batch vectors for
            each key in the list. (default: `None`)
        exclude_keys (list, optional): Will exclude each key in the list.
            (default: `None`)

    Returns:
        A tuple containing the collated data object and a dictionary
        holding optional batch feature masks.
    """
    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain an additional dictionary:
    # * `mask_dict` stores a mask representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    mask_dict = defaultdict(dict)
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():
            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == "num_nodes":
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors during dense collation:
            if attr == "ptr":
                continue

            # Collate attributes into a unified representation:
            value, mask = _dense_padded_collate(attr, values, data_list,
                                                stores)

            out_store[attr] = value
            if mask is not None:
                if key is not None:
                    mask_dict[key][attr] = mask
                else:
                    mask_dict[attr] = mask

    return out, mask_dict


def dense_padded_from_data_list(
    data_list: List[BaseData],
    follow_batch: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
):
    r"""Constructs a dense `~torch_geometric.data.Batch` object from a Python list of
    `~torch_geometric.data.Data` or `~torch_geometric.data.HeteroData` objects. The assignment
    vector `batch` is created on the fly. In addition, creates assignment vectors for each key in
    `follow_batch`. Will exclude any keys given in `exclude_keys`.

    Args:
        data_list (list): A list of data objects.
        follow_batch (list, optional): Creates assignment batch vectors for
            each key in the list. (default: `None`)
        exclude_keys (list, optional): Will exclude each key in the list.
            (default: `None`)

    Returns:
        A single `Batch` object holding a mini-batch of data.
    """
    batch, mask_dict = dense_padded_collate(
        Batch,
        data_list=data_list,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )

    batch._num_graphs = len(data_list)
    batch.mask_dict = mask_dict # NOTE: `mask_dict` must be a public attribute of `Batch` for auto-device onloading to work.

    return batch


class DensePaddingCollater:
    """Collates data objects from a `torch_geometric.data.Dataset` to a mini-batch using padding
    along with a new dimension.

    Data objects can be
    either of type `~torch_geometric.data.Data` or
    `~torch_geometric.data.HeteroData`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        """Collates a python list of data objects to the internal storage format of
        `torch_geometric.data.DataLoader`.

        Args:
            batch: A python list of data objects.

        Returns:
            A mini-batch of data objects.
        """
        elem = batch[0]
        if isinstance(elem, BaseData):
            return dense_padded_from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, along="row")
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        """Collates a python list of data objects to the internal storage format of
        `torch_geometric.data.DataLoader`.

        Args:
            batch: A python list of data objects.

        Returns:
            A mini-batch of data objects.
        """
        if isinstance(self.dataset, OnDiskDataset):
            return self(self.dataset.multi_get(batch))
        return self(batch)


class DensePaddingDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a `torch_geometric.data.Dataset` to a mini-
    batch using padding along with a new dimension. Data objects can be either of type
    `~torch_geometric.data.Data` or `~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: `1`)
        shuffle (bool, optional): If set to `True`, the data will be
            reshuffled at every epoch. (default: `False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: `None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: `None`)
        **kwargs (optional): Additional arguments of
            `torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = DensePaddingCollater(dataset, follow_batch,
                                             exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )

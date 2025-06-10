from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import EdgeIndex, Index
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.edge_index import SortOrder
from torch_geometric.typing import (
    SparseTensor,
    TensorFrame,
    torch_frame,
    torch_sparse,
)
from torch_geometric.utils import cumsum, is_sparse, is_torch_sparse_tensor
from torch_geometric.utils.sparse import cat

T = TypeVar('T')
SliceDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]
IncDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]


def collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:  # Dynamic inheritance.
        out = cls(_base_cls=data_list[0].__class__)  # type: ignore
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])  # type: ignore

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_stores = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device: Optional[torch.device] = None
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:  # type: ignore
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)

            # If parts of the data are already on GPU, make sure that auxiliary
            # data like `batch` or `ptr` are also created on GPU:
            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value

            if key is not None:  # Heterogeneous:
                store_slice_dict = slice_dict.get(key, {})
                assert isinstance(store_slice_dict, dict)
                store_slice_dict[attr] = slices
                slice_dict[key] = store_slice_dict

                store_inc_dict = inc_dict.get(key, {})
                assert isinstance(store_inc_dict, dict)
                store_inc_dict[attr] = incs
                inc_dict[key] = store_inc_dict
            else:  # Homogeneous:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f'{attr}_batch'] = batch
                out_store[f'{attr}_ptr'] = ptr

        # In case of node-level storages, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [store.num_nodes or 0 for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
) -> Tuple[Any, Any, Any]:

    elem = values[0]

    if isinstance(elem, Tensor) and not is_sparse(elem):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if cat_dim is None or elem.dim() == 0:
            values = [value.unsqueeze(0) for value in values]
        sizes = torch.tensor([value.size(cat_dim or 0) for value in values])
        slices = cumsum(sizes)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [
                    value + inc.to(value.device)
                    for value, inc in zip(values, incs)
                ]
        else:
            incs = None

        if getattr(elem, 'is_nested', False):
            tensors = []
            for nested_tensor in values:
                tensors.extend(nested_tensor.unbind())
            value = torch.nested.nested_tensor(tensors)

            return value, slices, incs

        out = None
        if (torch.utils.data.get_worker_info() is not None
                and not isinstance(elem, (Index, EdgeIndex))):
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            if torch_geometric.typing.WITH_PT20:
                storage = elem.untyped_storage()._new_shared(
                    numel * elem.element_size(), device=elem.device)
            else:
                storage = elem.storage()._new_shared(numel, device=elem.device)
            shape = list(elem.size())
            if cat_dim is None or elem.dim() == 0:
                shape = [len(values)] + shape
            else:
                shape[cat_dim] = int(slices[-1])
            out = elem.new(storage).resize_(*shape)

        value = torch.cat(values, dim=cat_dim or 0, out=out)

        if increment and isinstance(value, Index) and values[0].is_sorted:
            # Check whether the whole `Index` is sorted:
            if (value.diff() >= 0).all():
                value._is_sorted = True

        if increment and isinstance(value, EdgeIndex) and values[0].is_sorted:
            # Check whether the whole `EdgeIndex` is sorted by row:
            if values[0].is_sorted_by_row and (value[0].diff() >= 0).all():
                value._sort_order = SortOrder.ROW
            # Check whether the whole `EdgeIndex` is sorted by column:
            elif values[0].is_sorted_by_col and (value[1].diff() >= 0).all():
                value._sort_order = SortOrder.COL

        return value, slices, incs

    elif isinstance(elem, TensorFrame):
        key = str(key)
        sizes = torch.tensor([value.num_rows for value in values])
        slices = cumsum(sizes)
        value = torch_frame.cat(values, dim=0)
        return value, slices, None

    elif is_sparse(elem) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(torch.tensor(repeats))
        if is_torch_sparse_tensor(elem):
            value = cat(values, dim=cat_dim)
        else:
            value = torch_sparse.cat(values, dim=cat_dim)
        return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = torch.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment)
        return value_dict, slice_dict, inc_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str)
          and len(elem) > 0 and isinstance(elem[0], (Tensor, SparseTensor))):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(key, [v[i] for v in values],
                                           data_list, stores, increment)
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = torch.arange(len(values) + 1)
        return values, slices, None


def _batch_and_ptr(
    slices: Any,
    device: Optional[torch.device] = None,
) -> Tuple[Any, Any]:
    if (isinstance(slices, Tensor) and slices.dim() == 1):
        # Default case, turn slices tensor into batch.
        repeats = slices[1:] - slices[:-1]
        batch = repeat_interleave(repeats.tolist(), device=device)
        ptr = cumsum(repeats.to(device))
        return batch, ptr

    elif isinstance(slices, Mapping):
        # Recursively batch elements of dictionaries.
        batch, ptr = {}, {}
        for k, v in slices.items():
            batch[k], ptr[k] = _batch_and_ptr(v, device)
        return batch, ptr

    elif (isinstance(slices, Sequence) and not isinstance(slices, str)
          and isinstance(slices[0], Tensor)):
        # Recursively batch elements of lists.
        batch, ptr = [], []
        for s in slices:
            sub_batch, sub_ptr = _batch_and_ptr(s, device)
            batch.append(sub_batch)
            ptr.append(sub_ptr)
        return batch, ptr

    else:
        # Failure of batching, usually due to slices.dim() != 1
        return None, None


###############################################################################


def repeat_interleave(
    repeats: List[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    outs = [torch.full((n, ), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def get_incs(key, values: List[Any], data_list: List[BaseData],
             stores: List[BaseStorage]) -> Tensor:
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ]
    if isinstance(repeats[0], Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    return cumsum(repeats[:-1])

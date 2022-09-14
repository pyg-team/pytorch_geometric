from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage


def separate(cls, batch: BaseData, idx: int, slice_dict: Any,
             inc_dict: Any = None, decrement: bool = True) -> BaseData:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    data = cls().stores_as(batch)

    # We iterate over each storage object and recursively separate all its
    # attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        key = batch_store._key
        if key is not None:
            attrs = slice_dict[key].keys()
        else:
            attrs = set(batch_store.keys())
            attrs = [attr for attr in slice_dict.keys() if attr in attrs]
        for attr in attrs:
            if key is not None:
                slices = slice_dict[key][attr]
                incs = inc_dict[key][attr] if decrement else None
            else:
                slices = slice_dict[attr]
                incs = inc_dict[attr] if decrement else None
            data_store[attr] = _separate(attr, batch_store[attr], idx, slices,
                                         incs, batch, batch_store, decrement)

        # The `num_nodes` attribute needs special treatment, as we cannot infer
        # the real number of nodes from the total number of nodes alone:
        if hasattr(batch_store, '_num_nodes'):
            data_store.num_nodes = batch_store._num_nodes[idx]

    return data


def _separate(
    key: str,
    value: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
) -> Any:

    if isinstance(value, Tensor):
        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = value.narrow(cat_dim or 0, start, end - start)
        value = value.squeeze(0) if cat_dim is None else value
        if decrement and (incs.dim() > 1 or int(incs[idx]) != 0):
            value = value - incs[idx].to(value.device)
        return value

    elif isinstance(value, SparseTensor) and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        for i, dim in enumerate(cat_dims):
            start, end = int(slices[idx][i]), int(slices[idx + 1][i])
            value = value.narrow(dim, start, end - start)
        return value

    elif isinstance(value, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key: _separate(key, elem, idx, slices[key],
                           incs[key] if decrement else None, batch, store,
                           decrement)
            for key, elem in value.items()
        }

    elif (isinstance(value, Sequence) and isinstance(value[0], Sequence)
          and not isinstance(value[0], str) and len(value[0]) > 0
          and isinstance(value[0][0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of lists.
        return [elem[idx] for elem in value]

    elif (isinstance(value, Sequence) and not isinstance(value, str)
          and isinstance(value[0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of Tensors/SparseTensors.
        return [
            _separate(key, elem, idx, slices[i],
                      incs[i] if decrement else None, batch, store, decrement)
            for i, elem in enumerate(value)
        ]

    else:
        return value[idx]

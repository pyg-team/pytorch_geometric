from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar

from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage
from torch_geometric.typing import SparseTensor, TensorFrame
from torch_geometric.utils import narrow

T = TypeVar('T')


def separate(
    cls: Type[T],
    batch: Any,
    idx: int,
    slice_dict: Any,
    inc_dict: Any = None,
    decrement: bool = True,
) -> T:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    data = cls().stores_as(batch)

    # Iterate over each storage object and recursively separate its attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        key = batch_store._key
        if key is not None:  # Heterogeneous:
            attrs = slice_dict[key].keys()
        else:  # Homogeneous:
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
    values: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
) -> Any:

    if isinstance(values, Tensor):
        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, values, store)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = narrow(values, cat_dim or 0, start, end - start)
        value = value.squeeze(0) if cat_dim is None else value

        if isinstance(values, EdgeIndex) and values._cat_metadata is not None:
            # Reconstruct original `EdgeIndex` metadata:
            value._sparse_size = values._cat_metadata.sparse_size[idx]
            value._sort_order = values._cat_metadata.sort_order[idx]
            value._is_undirected = values._cat_metadata.is_undirected[idx]

        if (decrement and incs is not None
                and (incs.dim() > 1 or int(incs[idx]) != 0)):
            value = value - incs[idx].to(value.device)

        return value

    elif isinstance(values, SparseTensor) and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, values, store)
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        for i, dim in enumerate(cat_dims):
            start, end = int(slices[idx][i]), int(slices[idx + 1][i])
            values = values.narrow(dim, start, end - start)
        return values

    elif isinstance(values, TensorFrame):
        key = str(key)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = values[start:end]
        return value

    elif isinstance(values, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key:
            _separate(
                key,
                value,
                idx,
                slices=slices[key],
                incs=incs[key] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for key, value in values.items()
        }

    elif (isinstance(values, Sequence) and isinstance(values[0], Sequence)
          and not isinstance(values[0], str) and len(values[0]) > 0
          and isinstance(values[0][0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of lists.
        return [value[idx] for value in values]

    elif (isinstance(values, Sequence) and not isinstance(values, str)
          and isinstance(values[0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of Tensors/SparseTensors.
        return [
            _separate(
                key,
                value,
                idx,
                slices=slices[i],
                incs=incs[i] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            ) for i, value in enumerate(values)
        ]

    else:
        return values[idx]

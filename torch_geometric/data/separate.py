from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar

import torch
from torch import Tensor
from torch.nn.functional import pad

from torch_geometric import EdgeIndex, Index
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage
from torch_geometric.typing import SparseTensor, TensorFrame

T = TypeVar('T')


def separate(
    cls: Type[T],
    batch: BaseData,
    idx: torch.Tensor,
    slice_dict: Any,
    inc_dict: Any = None,
    decrement: bool = True,
    return_batch: bool = True,
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
        # TODO make less ugly
        if hasattr(batch_store, '_num_nodes'):
            if return_batch:
                data_store._num_nodes = [
                    batch_store._num_nodes[i] for i in idx
                ]
            else:
                data_store._num_nodes = batch_store._num_nodes[idx[0]]
            if hasattr(batch_store, 'num_nodes'):
                data_store.num_nodes = data_store._num_nodes

    return data


def _separate(
    key: str,
    values: Any,
    idx: torch.Tensor,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
) -> Any:
    if isinstance(values, Tensor):
        idx = idx.to(values.device)
        graph_slice = torch.concat([
            torch.arange(int(slices[i]), int(slices[i + 1])) for i in idx
        ]).to(values.device)
        valid_inc = incs is not None and (incs.dim() > 1
                                          or any(incs[idx] != 0))

        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, values, store)
        value = torch.index_select(values, cat_dim or 0, graph_slice)
        value = value.squeeze(0) if cat_dim is None else value

        if (decrement and incs is not None and valid_inc):
            # remove the old offset
            nelem_new = slices.diff()[idx]
            if len(idx) == 1:
                old_offset = incs[idx[0]]
                new_offset = torch.zeros_like(old_offset)
                shift = torch.ones_like(value) * (-old_offset + new_offset)
            else:
                old_offset = incs[idx]
                # add the new offset
                # for this we compute the number of nodes in the batch before
                new_offset = pad(incs.diff()[idx[:-1]], (1, 0)).cumsum(0)
                shift = (-old_offset + new_offset).repeat_interleave(
                    nelem_new, dim=cat_dim or 0)
            value = value + shift

        if hasattr(values,
                   "_cat_metadata") and values._cat_metadata is not None:

            def _pad_diff(a):
                a = torch.tensor(a)
                return torch.concat([a[:1], a.diff(dim=0)])

            if isinstance(values, Index):
                # Reconstruct original `Index` metadata:
                if decrement:
                    value._dim_size = _pad_diff(
                        values._cat_metadata.dim_size)[idx].squeeze()
                else:
                    value._dim_size = values._cat_metadata.dim_size[idx]
                value._is_sorted = values._cat_metadata.is_sorted[idx]

            if isinstance(values, EdgeIndex):
                # Reconstruct original `EdgeIndex` metadata:
                def _to_tup(a):
                    return tuple(a.flatten().tolist())

                if decrement:
                    value._sparse_size = _to_tup(
                        _pad_diff(values._cat_metadata.sparse_size)[idx])
                else:
                    value._sparse_size = values._cat_metadata.sparse_size[idx]
                value._sort_order = values._cat_metadata.sort_order[idx]
                value._is_undirected = values._cat_metadata.is_undirected[idx]

        return value

    elif isinstance(values, SparseTensor) and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        if len(idx) > 1:
            raise NotImplementedError
        idx = idx[0]

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
        values = values[start:end]
        return values

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
        if len(idx) > 1:
            raise NotImplementedError
        idx = idx[0]
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
    elif isinstance(values, list) and batch._num_graphs == len(values):
        if len(idx) == 1:
            return values[idx[0]]
        else:
            return [values[i] for i in idx]
    else:
        return values[idx]

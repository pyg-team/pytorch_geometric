import inspect
from collections.abc import Sequence
from typing import Any, List, Optional, Type, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.separate import separate


class DynamicInheritance(type):
    # A meta class that sets the base class of a `Batch` object, e.g.:
    # * `Batch(Data)` in case `Data` objects are batched together
    # * `Batch(HeteroData)` in case `HeteroData` objects are batched together
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        base_cls = kwargs.pop('_base_cls', Data)

        if issubclass(base_cls, Batch):
            new_cls = base_cls
        else:
            name = f'{base_cls.__name__}{cls.__name__}'

            # NOTE `MetaResolver` is necessary to resolve metaclass conflict
            # problems between `DynamicInheritance` and the metaclass of
            # `base_cls`. In particular, it creates a new common metaclass
            # from the defined metaclasses.
            class MetaResolver(type(cls), type(base_cls)):  # type: ignore
                pass

            if name not in globals():
                globals()[name] = MetaResolver(name, (cls, base_cls), {})
            new_cls = globals()[name]

        params = list(inspect.signature(base_cls.__init__).parameters.items())
        for i, (k, v) in enumerate(params[1:]):
            if k == 'args' or k == 'kwargs':
                continue
            if i < len(args) or k in kwargs:
                continue
            if v.default is not inspect.Parameter.empty:
                continue
            kwargs[k] = None

        return super(DynamicInheritance, new_cls).__call__(*args, **kwargs)


class DynamicInheritanceGetter:
    def __call__(self, cls: Type, base_cls: Type) -> Self:
        return cls(_base_cls=base_cls)


class Batch(metaclass=DynamicInheritance):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.

    :pyg:`PyG` allows modification to the underlying batching procedure by
    overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
    functionalities.
    The :meth:`~Data.__inc__` method defines the incremental count between two
    consecutive graph attributes.
    By default, :pyg:`PyG` increments attributes by the number of nodes
    whenever their attribute names contain the substring :obj:`index`
    (for historical reasons), which comes in handy for attributes such as
    :obj:`edge_index` or :obj:`node_index`.
    However, note that this may lead to unexpected behavior for attributes
    whose names contain the substring :obj:`index` but should not be
    incremented.
    To make sure, it is best practice to always double-check the output of
    batching.
    Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
    tensors of the same attribute should be concatenated together.
    """
    @classmethod
    def from_data_list(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)  # type: ignore
        batch._slice_dict = slice_dict  # type: ignore
        batch._inc_dict = inc_dict  # type: ignore

        return batch

    def get_example(self, idx: int) -> BaseData:
        r"""Gets the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object at index :obj:`idx`.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object.
        """
        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                "Cannot reconstruct 'Data' object from 'Batch' because "
                "'Batch' was not created via 'Batch.from_data_list()'")

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        return data

    def index_select(self, idx: IndexType) -> List[BaseData]:
        r"""Creates a subset of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from specified
        indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects.
        """
        index: Sequence[int]
        if isinstance(idx, slice):
            index = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            index = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            index = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            index = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            index = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            index = idx

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self.get_example(i) for i in index]

    def filter(self, idx: torch.Tensor) -> Self:
        """Efficiently filters Batch object using a boolean mask or index,
        directly modifying batch attributes instead of converting it to data
        list, masking, and rebuilding it (which can be ~10x slower).

        The provided indices (:obj:`idx`) can be a list, a tuple, a slicing
        object (e.g., :obj:`[2:5]`), or a :obj:`torch.Tensor`/:obj:`np.ndarray`
        of type long or bool, or any sequence of integers (excluding strings).
        """
        mask = torch.zeros(len(self), dtype=torch.bool)
        try:
            mask[idx] = True
        except IndexError as e:
            raise IndexError(
                "Invalid index provided. Ensure the index is "
                "compatible with PyTorch's indexing rules.") from e

        # Create new empty batch that will be filled later
        batch = Batch(_base_cls=self[0].__class__).stores_as(self)
        batch._slice_dict = {}
        batch._inc_dict = {}

        # Update the number of graphs based on the mask
        batch._num_graphs = mask.sum().item()

        # Return empty batch when mask filters all elements
        if batch._num_graphs == 0:
            return batch

        # Main loop to apply the mask at different levels (graph, nodes, edges)
        for old_store, new_store in zip(self.stores, batch.stores):
            # We get slices dictionary from key. If key is None then it means
            # we are dealing with graph level attributes.
            key = old_store._key

            if key is not None:  # Heterogeneous:
                attrs = self._slice_dict[key].keys()
            else:  # Homogeneous:
                attrs = set(old_store.keys())
                attrs = [
                    attr for attr in self._slice_dict.keys() if attr in attrs
                ]

            if key:
                batch._slice_dict[key] = {}
                batch._inc_dict[key] = {}

            # All slice and store are updated one by one in following loop
            for attr in attrs:
                if key is not None:
                    slc = self._slice_dict[key][attr]
                    incs = self._inc_dict[key][attr]
                else:
                    slc = self._slice_dict[attr]
                    incs = self._inc_dict[attr]

                slice_diff = slc.diff()

                # Reshape mask to align it with attribute shape. Since
                # slice_diff often contains only ones, skip useless
                # computation in such cases
                if torch.any(slice_diff != 1):
                    attr_mask = mask[torch.repeat_interleave(slice_diff)]
                else:
                    attr_mask = mask

                # Apply mask to attribute
                if attr == 'edge_index':
                    new_store[attr] = old_store[attr][:, attr_mask]
                elif isinstance(old_store[attr], list):
                    new_store[attr] = [
                        x for x, m in zip(old_store[attr], attr_mask) if m
                    ]
                else:
                    new_store[attr] = old_store[attr][attr_mask]

                # Compute masked version of slice tensor
                sizes_masked = slice_diff[mask]
                slice_masked = torch.cat(
                    (torch.zeros(1, dtype=torch.int), sizes_masked.cumsum(0)))

                # New _inc tensor is zeros tensor by default and can be
                # overwritten later if needed. For now, we only update it when
                # attr is edge_index, but we should do it every time original
                # _inc tensor is not zeros only.
                new_inc = torch.zeros(batch._num_graphs, dtype=torch.int)

                # when attr is 'x', we also update 'ptr' and 'batch' tensors
                # since this attribute provides node number information.
                if attr == 'x':
                    new_store['ptr'] = slice_masked
                    new_store['batch'] = torch.repeat_interleave(sizes_masked)

                # Reindex edge_index to remove gaps left by removed nodes
                if attr == 'edge_index':

                    # Compute diff tensor to get edge_index spans
                    old_spans = incs.diff(dim=0, append=incs[-1:])

                    # Apply the mask to filter spans
                    new_spans = old_spans[mask]

                    #  Use cumsum to reconstruct masked _inc tensor
                    new_inc_tmp = new_spans.cumsum(0)

                    # Adjust the result (start from zero, ignore last values)
                    new_inc_tmp[-1] = 0
                    new_inc = new_inc_tmp.roll(1, dims=0)

                    # Map each edge_index element to its batch position
                    attr_batch_map = torch.repeat_interleave(sizes_masked)

                    # Update edge_index by removing old_inc and add new_inc
                    # We do new_inc - old_inc operation before applying the
                    # map for efficiency purpose
                    shift = (new_inc - incs[mask])[attr_batch_map]

                    if shift.ndim == 1:  # Homogeneous
                        new_store[attr] += shift
                    else:  # Heterogeneous
                        new_store[attr] += shift.squeeze(-1).T

                # Finally, we update _slice_dict and _inc_dict based on what
                # has been computed in previous steps
                if key:  # Node or edge level attribute
                    batch._slice_dict[key][attr] = slice_masked
                    batch._inc_dict[key][attr] = new_inc
                else:  # Graph level attribute
                    batch._slice_dict[attr] = slice_masked
                    batch._inc_dict[attr] = new_inc
        return batch

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)  # type: ignore
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)  # type: ignore
        else:
            return self.filter(idx)

    def to_data_list(self) -> List[BaseData]:
        r"""Reconstructs the list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from the
        :class:`~torch_geometric.data.Batch` object.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects.
        """
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'ptr'):
            return self.ptr.numel() - 1
        elif hasattr(self, 'batch'):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    @property
    def batch_size(self) -> int:
        r"""Alias for :obj:`num_graphs`."""
        return self.num_graphs

    def __len__(self) -> int:
        return self.num_graphs

    def __reduce__(self) -> Any:
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state

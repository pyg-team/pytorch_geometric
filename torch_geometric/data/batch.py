import inspect
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import LongTensor, Tensor
from torch.nn.functional import pad
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
    def from_empty(cls, num_nodes: Union[Tensor, Sequence]) -> Self:
        if isinstance(num_nodes, Sequence):
            num_nodes = torch.tensor(num_nodes, dtype=torch.long)
        if not num_nodes.dtype == torch.long:
            raise Exception('`num_nodes` dtype must be torch.long')
        if not num_nodes.dim() == 1:
            raise Exception('`num_nodes` must have one dimension')
        if (num_nodes < 0).any():
            raise Exception('`num_nodes` must be positive')

        batch = Batch()
        batch.batch, batch.ptr = cls._batch_ptr_from_num_nodes(cls, num_nodes)
        batch._num_graphs = int(batch.batch.max() + 1)
        batch._slice_dict = defaultdict(dict)
        batch._inc_dict = defaultdict(dict)
        return batch

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
        The assignment vector :obj:`batch` is adjusted on the fly.
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

    @classmethod
    def from_batch_list(cls, batches: List[Self]) -> Self:
        r"""Same as :meth:`~Batch.from_data_list```,
        but for concatenating existing batches.
        Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Batch` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        batch = cls.from_data_list(batches)

        del batch._slice_dict['batch'], batch._inc_dict['batch']

        batch.batch, batch.ptr = cls._batch_ptr_from_num_nodes(
            cls, torch.concat([g.ptr.diff() for g in batches]))

        for k in set(batch.keys()) - {'batch', 'ptr'}:
            batch._slice_dict[k] = batch._pad_zero(
                torch.concat([be._slice_dict[k].diff()
                              for be in batches]).cumsum(0))
            if k != 'edge_index':
                inc_shift = batch._pad_zero(
                    torch.tensor([sum(be._inc_dict[k])
                                  for be in batches])).cumsum(0)
            else:
                inc_shift = batch._pad_zero(
                    torch.tensor([be.num_nodes for be in batches])).cumsum(0)

            batch._inc_dict[k] = torch.cat([
                be._inc_dict[k] + inc_shift[ibatch]
                for ibatch, be in enumerate(batches)
            ])
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
            idx=torch.tensor([idx]).long(),
            slice_dict=getattr(self, '_slice_dict'),
            inc_dict=getattr(self, '_inc_dict'),
            decrement=True,
            return_batch=False,
        )

        return data

    def index_select(self, idx: IndexType) -> Self:
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
        if not isinstance(idx, (slice, Sequence, torch.Tensor, np.ndarray)):
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f'np.ndarray of dtype long or bool are valid indices (got '
                f"'{type(idx).__name__}')")

        index: torch.Tensor

        def _to_tidx(o):
            return torch.tensor(o, device=self.ptr.device)

        # convert numpt to torch tensors
        if isinstance(idx, np.ndarray):
            idx = _to_tidx(idx)
        if isinstance(idx, slice):
            index = _to_tidx(range(self.num_graphs)[idx]).long()
        elif isinstance(idx, Sequence):
            index = _to_tidx(idx).long()
        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            index = idx
        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            if not len(idx) == self.num_graphs:
                IndexError(
                    f'Boolen vector length does not match number of graphs'
                    f' (got {len(idx)} vector size '
                    f'vs. {self.num_graphs} graphs).')
            index = idx.nonzero().flatten()
        else:
            raise IndexError(
                f"Could not convert index (got '{type(idx).__name__}')")

        if index.dim() != 1:
            raise IndexError(
                f'Index must have a single dimension (got {index.dim()})')

        self.ptr.device

        subbatch = separate(
            cls=self.__class__,
            batch=self,
            idx=index,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        nodes_per_graph = self.ptr.diff()
        num_nodes = nodes_per_graph[index]

        subbatch.batch, subbatch.ptr = self._batch_ptr_from_num_nodes(
            num_nodes)

        # fix the _slice_dict and _inc_dict
        subbatch._slice_dict = defaultdict(dict)
        subbatch._inc_dict = defaultdict(dict)
        for k in set(self.keys()) - {'ptr', 'batch'}:
            if k not in self._slice_dict:
                continue
            subbatch._slice_dict[k] = pad(self._slice_dict[k].diff()[index],
                                          (1, 0)).cumsum(0)
            if k not in self._inc_dict:
                continue
            if self._inc_dict[k] is None:
                subbatch._inc_dict[k] = None
                continue
            subbatch._inc_dict[k] = pad(self._inc_dict[k].diff()[index[:-1]],
                                        (1, 0)).cumsum(0)
        return subbatch

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        # Return single Graph
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)  # type: ignore
        # Return stored objects
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)  # type: ignore
        # Return subset of the batch
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[BaseData]:
        r"""Reconstructs the list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from the
        :class:`~torch_geometric.data.Batch` object.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects.
        """
        return [self.get_example(i) for i in range(self.num_graphs)]

    def set_attr(self, attrname: str, attr: Tensor,
                 attrtype: Literal["node", "graph", "edge"] = "node") -> None:
        r"""Set an attribute for the nodes, graphs, or edges.

        Args:
            attrname (str): Name of the attribute.
            attr (torch.Tensor): The attribute tensor.
            attrtype (str): Indicates if the attribution belongs to the nodes,
                graphs or edges (`node`, `graph`, `edge`), default: `node`.

        """
        if attrname == "edge_attr" and attrtype != "edge":
            raise Exception(
                "For the attribute `edge_attr`, the `attrtype` must be `edge`."
            )

        if attrname == "edge_index":
            raise Exception(
                """To overwrite the edges, use `set_edge_index`.""")

        assert attrtype in ["node", "graph", "edge"]

        assert attr.device == self.batch.device
        batch_idxs = self.batch

        self[attrname] = attr

        if attrtype == "node":
            assert attr.shape[0] == self.num_nodes
            out = batch_idxs.unique(return_counts=True)[1]
            out = out.cumsum(dim=0)
            self._slice_dict[attrname] = self._pad_zero(out).cpu()

            self._inc_dict[attrname] = torch.zeros(self._num_graphs,
                                                   dtype=torch.long)
        elif attrtype == "graph":
            assert attr.shape[0] == self.num_graphs
            self._slice_dict[attrname] = torch.arange(self.num_graphs + 1,
                                                      dtype=torch.long)
            self._inc_dict[attrname] = torch.zeros(self.num_graphs,
                                                   dtype=torch.long)
        elif attrtype == "edge":
            assert attr.shape[0] == self.num_edges
            assert (hasattr(self, 'edge_index')
                    and self['edge_index'].dtype == torch.long)
            self._slice_dict['edge_attr'] = self._slice_dict['edge_index']
            self._inc_dict['edge_attr'] = torch.zeros(self.num_graphs)
        else:
            raise NotImplementedError()

    def set_edge_index(self, edge_index: Union[List[LongTensor], LongTensor],
                       batchidx_per_edge: Optional[LongTensor] = None) -> None:
        r"""Overwrites the :obj:`edge_index`.
        :obj:`~Batch.ptr` will be used to assign the elements to
        the correct graph.

        Args:
            edge_index (Union[List[LongTensor], LongTensor]): Either a
                list of the new edges for each graph, or a tensor containing
                the new edges. In the latter case the assignment to the graphs
                must be give by `batchidx_per_edge`.
            batchidx_per_edge (Optional[LongTensor]): The index tensor
                that maps each of the edges to a graph.
        """
        if isinstance(edge_index, list):
            device = edge_index[0].device
            edges_per_graph = torch.tensor([e.shape[1] for e in edge_index])
            batchidx_per_edge = torch.arange(
                self.num_graphs).repeat_interleave(edges_per_graph).to(device)
            edge_index = torch.hstack(edge_index)
        else:
            edges_per_graph = batchidx_per_edge.unique(return_counts=True)[1]

            assert (batchidx_per_edge.diff()
                    >= 0).all(), 'Edges must be ordered by batch'
            assert batchidx_per_edge.shape == torch.Size(
                (edge_index.shape[1], ))

            assert edge_index.dim() == 2
            assert edge_index.shape[0] == 2

        assert edge_index.dtype == batchidx_per_edge.dtype == torch.long
        assert (edge_index.device == batchidx_per_edge.device ==
                self.batch.device)

        # Edges must be shifted by the number sum of the nodes
        # in the previous graphs
        self.edge_index = edge_index + self.ptr[batchidx_per_edge]
        # Fix _slice_dict
        self._slice_dict['edge_index'] = self._pad_zero(
            edges_per_graph.cumsum(0)).cpu()
        self._inc_dict['edge_index'] = self.ptr[:-1].cpu()

    def _pad_zero(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            torch.tensor(0, dtype=arr.dtype, device=arr.device).unsqueeze(0),
            arr,
        ])

    def _batch_ptr_from_num_nodes(
            self, num_nodes: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not num_nodes.dtype == torch.long:
            raise TypeError("Argument must have type `torch.long`")
        batch = torch.arange(
            len(num_nodes),
            device=num_nodes.device).repeat_interleave(num_nodes)
        ptr = pad(num_nodes, (1, 0), "constant", 0).cumsum(0)
        return batch, ptr

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
            raise ValueError('Can not infer the number of graphs')

    @property
    def batch_size(self) -> int:
        r"""Alias for :obj:`num_graphs`."""
        return self.num_graphs

    def __len__(self) -> int:
        return self.num_graphs

    def __reduce__(self) -> Any:
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state

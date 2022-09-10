from collections.abc import Sequence
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import (
    edge_type_to_str,
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    to_csc,
    to_hetero_csc,
)
from torch_geometric.typing import InputNodes, NumNeighbors


class NeighborSampler:
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        input_type: Optional[Any] = None,
        time_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        self.data_cls = data.__class__ if isinstance(
            data, (Data, HeteroData)) else 'custom'
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.node_time = None

        # TODO Unify the following conditionals behind the `FeatureStore`
        # and `GraphStore` API

        # If we are working with a `Data` object, convert the edge_index to
        # CSC and store it:
        if isinstance(data, Data):
            if time_attr is not None:
                # TODO `time_attr` support for homogeneous graphs
                raise ValueError(
                    f"'time_attr' attribute not yet supported for "
                    f"'{data.__class__.__name__}' object")

            # Convert the graph data into a suitable format for sampling.
            out = to_csc(data, device='cpu', share_memory=share_memory,
                         is_sorted=is_sorted)
            self.colptr, self.row, self.perm = out
            assert isinstance(num_neighbors, (list, tuple))

        # If we are working with a `HeteroData` object, convert each edge
        # type's edge_index to CSC and store it:
        elif isinstance(data, HeteroData):
            if time_attr is not None:
                self.node_time_dict = data.collect(time_attr)
            else:
                self.node_time_dict = None

            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data, device='cpu', share_memory=share_memory,
                                is_sorted=is_sorted)
            self.colptr_dict, self.row_dict, self.perm_dict = out

            self.node_types, self.edge_types = data.metadata()
            self._set_num_neighbors_and_num_hops(num_neighbors)

            assert input_type is not None
            self.input_type = input_type

        # If we are working with a `Tuple[FeatureStore, GraphStore]` object,
        # obtain edges from GraphStore and convert them to CSC if necessary,
        # storing the resulting representations:
        elif isinstance(data, tuple):
            # TODO support `FeatureStore` with no edge types (e.g. `Data`)
            feature_store, graph_store = data

            # TODO support `collect` on `FeatureStore`
            self.node_time_dict = None
            if time_attr is not None:
                # We need to obtain all features with 'attr_name=time_attr'
                # from the feature store and store them in node_time_dict. To
                # do so, we make an explicit feature store GET call here with
                # the relevant 'TensorAttr's
                time_attrs = [
                    attr for attr in feature_store.get_all_tensor_attrs()
                    if attr.attr_name == time_attr
                ]
                for attr in time_attrs:
                    attr.index = None
                time_tensors = feature_store.multi_get_tensor(time_attrs)
                self.node_time_dict = {
                    time_attr.group_name: time_tensor
                    for time_attr, time_tensor in zip(time_attrs, time_tensors)
                }

            # Obtain all node and edge metadata:
            node_attrs = feature_store.get_all_tensor_attrs()
            edge_attrs = graph_store.get_all_edge_attrs()

            self.node_types = list(
                set(node_attr.group_name for node_attr in node_attrs))
            self.edge_types = list(
                set(edge_attr.edge_type for edge_attr in edge_attrs))

            # Set other required parameters:
            self._set_num_neighbors_and_num_hops(num_neighbors)

            assert input_type is not None
            self.input_type = input_type

            # Obtain CSC representations for in-memory sampling:
            row_dict, colptr_dict, perm_dict = graph_store.csc()
            self.row_dict = {
                edge_type_to_str(k): v
                for k, v in row_dict.items()
            }
            self.colptr_dict = {
                edge_type_to_str(k): v
                for k, v in colptr_dict.items()
            }
            self.perm_dict = {
                edge_type_to_str(k): v
                for k, v in perm_dict.items()
            }

        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

    def _set_num_neighbors_and_num_hops(self, num_neighbors):
        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.edge_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = {
            edge_type_to_str(key): value
            for key, value in num_neighbors.items()
        }
        # Add at least one element to the list to ensure `max` is well-defined
        self.num_hops = max([0] + [len(v) for v in num_neighbors.values()])

    def _sparse_neighbor_sample(self, index: Tensor):
        fn = torch.ops.torch_sparse.neighbor_sample
        node, row, col, edge = fn(
            self.colptr,
            self.row,
            index,
            self.num_neighbors,
            self.replace,
            self.directed,
        )
        return node, row, col, edge

    def _hetero_sparse_neighbor_sample(self, index_dict: Dict[str, Tensor],
                                       **kwargs):
        if self.node_time_dict is None:
            fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                index_dict,
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
        else:
            try:
                fn = torch.ops.torch_sparse.hetero_temporal_neighbor_sample
            except RuntimeError as e:
                raise RuntimeError(
                    "'torch_sparse' operator "
                    "'hetero_temporal_neighbor_sample' not "
                    "found. Currently requires building "
                    "'torch_sparse' from master.", e)

            node_dict, row_dict, col_dict, edge_dict = fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                index_dict,
                self.num_neighbors,
                kwargs.get('node_time_dict', self.node_time_dict),
                self.num_hops,
                self.replace,
                self.directed,
            )
        return node_dict, row_dict, col_dict, edge_dict

    def __call__(self, index: Union[List[int], Tensor]):
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)

        if self.data_cls != 'custom' and issubclass(self.data_cls, Data):
            return self._sparse_neighbor_sample(index) + (index.numel(), )

        elif self.data_cls == 'custom' or issubclass(self.data_cls,
                                                     HeteroData):
            return self._hetero_sparse_neighbor_sample(
                {self.input_type: index}) + (index.numel(), )


class NeighborLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, up to :obj:`num_neighbors`
    neighbors will be sampled for each :obj:`edge_type`.
    However, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible:

    .. code-block:: python

        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader

        hetero_data = OGB_MAG(path)[0]

        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    The :class:`~torch_geometric.loader.NeighborLoader` will return subgraphs
    where global node indices are mapped to local indices corresponding to this
    specific subgraph. However, often times it is desired to map the nodes of
    the current subgraph back to the global node indices. A simple trick to
    achieve this is to include this mapping as part of the :obj:`data` object:

    .. code-block:: python

        # Assign each node its global node index:
        data.n_id = torch.arange(data.num_nodes)

        loader = NeighborLoader(data, ...)
        sampled_data = next(iter(loader))
        print(sampled_data.n_id)

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier timestamp than the center node. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighbors,
        input_nodes: InputNodes = None,
        replace: bool = False,
        directed: bool = True,
        time_attr: Optional[str] = None,
        transform: Callable = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        neighbor_sampler: Optional[NeighborSampler] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neighbor_sampler = neighbor_sampler

        node_type, input_nodes = get_input_nodes(data, input_nodes)

        if neighbor_sampler is None:
            self.neighbor_sampler = NeighborSampler(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=node_type,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )

        super().__init__(input_nodes, collate_fn=self.collate_fn, **kwargs)

    def filter_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            node, row, col, edge, batch_size = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)
            data.batch_size = batch_size

        elif isinstance(self.data, HeteroData):
            node_dict, row_dict, col_dict, edge_dict, batch_size = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)
            data[self.neighbor_sampler.input_type].batch_size = batch_size

        else:  # Tuple[FeatureStore, GraphStore]
            # TODO support for feature stores with no edge types
            node_dict, row_dict, col_dict, edge_dict, batch_size = out
            feature_store, graph_store = self.data
            data = filter_custom_store(feature_store, graph_store, node_dict,
                                       row_dict, col_dict, edge_dict)
            data[self.neighbor_sampler.input_type].batch_size = batch_size

        return data if self.transform is None else self.transform(data)

    def collate_fn(self, index: Union[List[int], Tensor]) -> Any:
        out = self.neighbor_sampler(index)
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        # We execute `filter_fn` in the main process.
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def get_input_nodes(
    data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
    input_nodes: Union[InputNodes, TensorAttr],
) -> Tuple[Optional[str], Sequence]:
    def to_index(tensor):
        if isinstance(tensor, Tensor) and tensor.dtype == torch.bool:
            return tensor.nonzero(as_tuple=False).view(-1)
        return tensor

    if isinstance(data, Data):
        if input_nodes is None:
            return None, range(data.num_nodes)
        return None, to_index(input_nodes)

    elif isinstance(data, HeteroData):
        assert input_nodes is not None

        if isinstance(input_nodes, str):
            return input_nodes, range(data[input_nodes].num_nodes)

        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)

        node_type, input_nodes = input_nodes
        if input_nodes is None:
            return node_type, range(data[node_type].num_nodes)
        return node_type, to_index(input_nodes)

    else:  # Tuple[FeatureStore, GraphStore]
        # NOTE FeatureStore and GraphStore are treated as separate
        # entities, so we cannot leverage the custom structure in Data and
        # HeteroData to infer the number of nodes. As a result, here we expect
        # that the input nodes are either explicitly provided or can be
        # directly inferred from the feature store.
        feature_store, _ = data

        assert input_nodes is not None

        if isinstance(input_nodes, Tensor):
            return None, to_index(input_nodes)

        # Can't infer number of nodes from a group_name; need an attr_name
        if isinstance(input_nodes, str):
            raise NotImplementedError(
                f"Cannot infer the number of nodes from a single string "
                f"(got '{input_nodes}'). Please pass a more explicit "
                f"representation. ")

        if isinstance(input_nodes, (list, tuple)):
            assert len(input_nodes) == 2
            assert isinstance(input_nodes[0], str)

            node_type, input_nodes = input_nodes
            if input_nodes is None:
                raise NotImplementedError(
                    f"Cannot infer the number of nodes from a node type alone "
                    f"(got '{input_nodes}'). Please pass a more explicit "
                    f"representation. ")
            return node_type, to_index(input_nodes)

        assert isinstance(input_nodes, TensorAttr)
        assert input_nodes.is_set('attr_name')

        node_type = getattr(input_nodes, 'group_name', None)
        if not input_nodes.is_set('index') or input_nodes.index is None:
            num_nodes = feature_store.get_tensor_size(input_nodes)[0]
            return node_type, range(num_nodes)
        return node_type, input_nodes.index
